package kvbudget

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"net/http/httputil"
	"net/url"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// fakeEngine is the llama.cpp surface this gate depends on: the two tokenizer
// hops, plus a chat route slow enough that a second request arrives while the
// first still holds its reservation.
type fakeEngine struct {
	// tokensPerCall is what /tokenize reports, whatever it was given.
	tokensPerCall int

	// chatDelay holds a chat request open, standing in for generation.
	chatDelay time.Duration

	// tokenizeStatus, when non-zero, is returned by /tokenize instead of a
	// count.
	tokenizeStatus int

	chatHits     atomic.Int32
	tokenizeHits atomic.Int32
	templateHits atomic.Int32

	// release, when non-nil, gates chat completion instead of chatDelay.
	release chan struct{}
}

func (f *fakeEngine) server(t *testing.T) *httptest.Server {
	t.Helper()
	mux := http.NewServeMux()
	mux.HandleFunc("/apply-template", func(w http.ResponseWriter, r *http.Request) {
		f.templateHits.Add(1)
		var in struct {
			Messages []struct {
				Content string `json:"content"`
			} `json:"messages"`
		}
		_ = json.NewDecoder(r.Body).Decode(&in)
		var b strings.Builder
		for _, m := range in.Messages {
			b.WriteString(m.Content)
		}
		writeJSON(w, map[string]any{"prompt": b.String()})
	})
	mux.HandleFunc("/tokenize", func(w http.ResponseWriter, r *http.Request) {
		f.tokenizeHits.Add(1)
		if f.tokenizeStatus != 0 {
			w.WriteHeader(f.tokenizeStatus)
			return
		}
		ids := make([]int, f.tokensPerCall)
		writeJSON(w, map[string]any{"tokens": ids})
	})
	mux.HandleFunc("/v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		f.chatHits.Add(1)
		if f.release != nil {
			<-f.release
		} else if f.chatDelay > 0 {
			time.Sleep(f.chatDelay)
		}
		writeJSON(w, map[string]any{"ok": true})
	})
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)
	return srv
}

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(v)
}

// chatBody builds a chat request whose prompt is padded to n bytes, so a test
// can put the byte bound where it wants it relative to the pool.
func chatBody(n int) string {
	return `{"messages":[{"role":"user","content":"` + strings.Repeat("x", n) + `"}]}`
}

func postChat(t *testing.T, h http.Handler, body string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	return rec
}

// TestSecondLargeRequestIsRefusedWithoutReachingTheEngine is the acceptance
// case: the failure this package exists to prevent is two large requests
// overlapping, and the proof it works is that the second one never arrives at
// the engine at all.
func TestSecondLargeRequestIsRefusedWithoutReachingTheEngine(t *testing.T) {
	engine := &fakeEngine{tokensPerCall: 7000, release: make(chan struct{})}
	srv := engine.server(t)

	g := New(Options{
		Capacity:        func() (int, bool) { return 12288, true },
		Tokenizer:       &Tokenizer{BaseURL: srv.URL},
		MaxOutputTokens: func() int { return 4096 },
		Wait:            150 * time.Millisecond,
	})
	h := g.Wrap(proxyTo(t, srv.URL))

	// 7000 prompt + 2048 default output = 9048 of a 12288 pool. Two of
	// those do not fit, and the body is well past an eighth of the pool so
	// both are measured rather than fast-pathed.
	body := chatBody(4000)

	first := make(chan int, 1)
	go func() { first <- postChat(t, h, body).Code }()

	// Wait for the first to be holding its reservation inside the engine.
	waitFor(t, func() bool { return engine.chatHits.Load() == 1 })

	second := postChat(t, h, body)
	if second.Code != http.StatusServiceUnavailable {
		t.Fatalf("second request status = %d, want 503", second.Code)
	}
	if got := engine.chatHits.Load(); got != 1 {
		t.Errorf("engine saw %d chat requests; the refused one must not reach it", got)
	}
	assertExhausted(t, second)

	close(engine.release)
	if code := <-first; code != http.StatusOK {
		t.Errorf("first request status = %d, want 200", code)
	}
}

// TestAGuardStartsAccountingWhenTheFlagsChangeUnderIt is the same acceptance
// case, with the deployment arriving at the configuration rather than booting
// into it.
//
// The flags are on an editable card: a deployment that started with one slot
// is relaunched sharing a pool between several the moment somebody removes
// -np. A gate that answered "does this apply to me" once, while the process
// was assembling itself, stayed out of the way for the rest of that process's
// life -- so the overlap below failed with the engine's 500 instead.
func TestAGuardStartsAccountingWhenTheFlagsChangeUnderIt(t *testing.T) {
	engine := &fakeEngine{tokensPerCall: 7000, release: make(chan struct{})}
	srv := engine.server(t)

	wanted := false
	g := New(Options{
		Active:          func() bool { return wanted },
		Capacity:        func() (int, bool) { return 12288, true },
		Tokenizer:       &Tokenizer{BaseURL: srv.URL},
		MaxOutputTokens: func() int { return 4096 },
		Wait:            150 * time.Millisecond,
	})
	h := g.Wrap(proxyTo(t, srv.URL))

	// 7000 prompt + 4096 output = 11096 of a 12288 pool, so no two of these
	// fit at once once anything is counting them.
	body := chatBody(4000)

	held := make(chan int, 3)
	post := func() { held <- postChat(t, h, body).Code }

	// While the flags do not describe a shared pool, two of these overlap
	// untouched: each slot owns a hard share the engine enforces itself.
	go post()
	go post()
	waitFor(t, func() bool { return engine.chatHits.Load() == 2 })

	// What an operator does through the Model Console: the card is edited to
	// drop -np, and the engine is relaunched sharing one pool between its
	// slots. Nothing about this process restarted.
	wanted = true

	go post()
	waitFor(t, func() bool { return engine.chatHits.Load() == 3 })

	refused := postChat(t, h, body)
	if refused.Code != http.StatusServiceUnavailable {
		t.Fatalf("request status = %d after the flags changed, want 503", refused.Code)
	}
	if got := engine.chatHits.Load(); got != 3 {
		t.Errorf("engine saw %d chat requests; the refused one must not reach it", got)
	}
	assertExhausted(t, refused)

	close(engine.release)
	for i := range 3 {
		if code := <-held; code != http.StatusOK {
			t.Errorf("held request %d status = %d, want 200", i, code)
		}
	}
}

func assertExhausted(t *testing.T, rec *httptest.ResponseRecorder) {
	t.Helper()
	if got := rec.Header().Get("Retry-After"); got == "" || got == "0" {
		t.Errorf("Retry-After = %q, want a positive whole number of seconds", got)
	}
	var env struct {
		Error struct {
			Code    string `json:"code"`
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &env); err != nil {
		t.Fatalf("decode 503 envelope: %v (body %q)", err, rec.Body.String())
	}
	if env.Error.Code != CodeExhausted {
		t.Errorf("error.code = %q, want %q", env.Error.Code, CodeExhausted)
	}
	if env.Error.Message == "" {
		t.Error("error.message is empty; the 503 has to say what happened")
	}
}

// TestRoomIsReleasedWhenTheRequestEnds covers the half of the accounting that
// a semaphore cannot report on: a reservation that is never released turns
// the first burst of traffic into a permanent outage.
func TestRoomIsReleasedWhenTheRequestEnds(t *testing.T) {
	engine := &fakeEngine{tokensPerCall: 7000}
	srv := engine.server(t)

	g := New(Options{
		Capacity:  func() (int, bool) { return 12288, true },
		Tokenizer: &Tokenizer{BaseURL: srv.URL},
		Wait:      time.Second,
	})
	h := g.Wrap(proxyTo(t, srv.URL))

	for i := range 3 {
		if rec := postChat(t, h, chatBody(4000)); rec.Code != http.StatusOK {
			t.Fatalf("request %d status = %d, want 200: each should reuse the room the previous one gave back", i+1, rec.Code)
		}
	}
}

// TestSmallRequestsSkipTheTokenizer is the short-circuit: a request that
// cannot possibly matter must not pay two round trips to prove it.
func TestSmallRequestsSkipTheTokenizer(t *testing.T) {
	engine := &fakeEngine{tokensPerCall: 10}
	srv := engine.server(t)

	g := New(Options{
		Capacity:  func() (int, bool) { return 131072, true },
		Tokenizer: &Tokenizer{BaseURL: srv.URL},
		// 2048 default output + a short body is well under 131072/8.
	})
	h := g.Wrap(proxyTo(t, srv.URL))

	if rec := postChat(t, h, chatBody(50)); rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", rec.Code)
	}
	if got := engine.tokenizeHits.Load(); got != 0 {
		t.Errorf("/tokenize was called %d times for a request that fits eight times over", got)
	}
	if got := engine.templateHits.Load(); got != 0 {
		t.Errorf("/apply-template was called %d times for the same request", got)
	}
}

// TestLargeRequestsAreMeasured is the other side of it: past the fast path the
// gate pays for a real count, which is the only way the charge can be smaller
// than the byte bound.
func TestLargeRequestsAreMeasured(t *testing.T) {
	engine := &fakeEngine{tokensPerCall: 100}
	srv := engine.server(t)

	g := New(Options{
		Capacity:  func() (int, bool) { return 4096, true },
		Tokenizer: &Tokenizer{BaseURL: srv.URL},
	})
	h := g.Wrap(proxyTo(t, srv.URL))

	// 3000 body bytes bound to 3000 + 2048 = 5048, over a 4096 pool. Only a
	// real count (100 + 2048) lets this through at all.
	if rec := postChat(t, h, chatBody(3000)); rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", rec.Code)
	}
	if got := engine.templateHits.Load(); got != 1 {
		t.Errorf("/apply-template hits = %d, want 1", got)
	}
	if got := engine.tokenizeHits.Load(); got != 1 {
		t.Errorf("/tokenize hits = %d, want 1", got)
	}
}

// TestTokenizeFailureStillServes is the degradation rule: the tokenizer is an
// accuracy aid, and losing it must not cost the model its ability to answer.
func TestTokenizeFailureStillServes(t *testing.T) {
	engine := &fakeEngine{tokenizeStatus: http.StatusInternalServerError}
	srv := engine.server(t)

	g := New(Options{
		Capacity:  func() (int, bool) { return 131072, true },
		Tokenizer: &Tokenizer{BaseURL: srv.URL},
	})
	h := g.Wrap(proxyTo(t, srv.URL))

	// Past the fast path, so the count is attempted and fails; the bound
	// stands in for it and still fits.
	if rec := postChat(t, h, chatBody(20000)); rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200: a broken tokenizer must not refuse service", rec.Code)
	}
	if got := engine.chatHits.Load(); got != 1 {
		t.Errorf("engine saw %d chat requests, want 1", got)
	}
}

// TestRequestLargerThanThePoolIsForwarded covers the case waiting cannot fix.
// The engine's refusal names the real limit; a 503 from here would tell the
// caller to retry something that can never succeed.
func TestRequestLargerThanThePoolIsForwarded(t *testing.T) {
	engine := &fakeEngine{tokensPerCall: 90000}
	srv := engine.server(t)

	g := New(Options{
		Capacity:  func() (int, bool) { return 4096, true },
		Tokenizer: &Tokenizer{BaseURL: srv.URL},
		Wait:      50 * time.Millisecond,
	})
	h := g.Wrap(proxyTo(t, srv.URL))

	if rec := postChat(t, h, chatBody(10000)); rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want the engine's own answer", rec.Code)
	}
	if got := engine.chatHits.Load(); got != 1 {
		t.Errorf("engine saw %d chat requests; an unfittable request still has to reach it", got)
	}
}

// TestNoCapacityMeansNoGate is the rule that keeps this package from being a
// liability on a deployment nobody could measure.
func TestNoCapacityMeansNoGate(t *testing.T) {
	engine := &fakeEngine{tokensPerCall: 90000}
	srv := engine.server(t)

	g := New(Options{
		Capacity:  func() (int, bool) { return 0, false },
		Tokenizer: &Tokenizer{BaseURL: srv.URL},
	})
	h := g.Wrap(proxyTo(t, srv.URL))

	for range 3 {
		if rec := postChat(t, h, chatBody(50000)); rec.Code != http.StatusOK {
			t.Fatalf("status = %d, want 200: with no pool size there is nothing to account against", rec.Code)
		}
	}
	if got := engine.tokenizeHits.Load(); got != 0 {
		t.Errorf("/tokenize was called %d times with the gate inactive", got)
	}
}

// TestBodylessRequestsReserveNothing is why the gate reads the body rather
// than the path. It is mounted on the `/v1/` catch-all, so a metadata call
// passes through it, and charging one for a completion it will never generate
// takes pool room away from requests that need it.
func TestBodylessRequestsReserveNothing(t *testing.T) {
	engine := &fakeEngine{tokensPerCall: 10}
	srv := engine.server(t)

	// A pool small enough that a single default output reserve would fill
	// it, so an overcharged metadata call could not hide.
	const pool = defaultOutputReserve
	g := New(Options{
		Capacity:  func() (int, bool) { return pool, true },
		Tokenizer: &Tokenizer{BaseURL: srv.URL},
		Wait:      50 * time.Millisecond,
	})

	// Concurrency is the assertion, not the status code: four calls that
	// each reserved the whole pool would still all return 200, just one
	// after another. Only their overlap says nothing was reserved.
	var inFlight, peak atomic.Int32
	h := g.Wrap(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n := inFlight.Add(1)
		for {
			old := peak.Load()
			if n <= old || peak.CompareAndSwap(old, n) {
				break
			}
		}
		defer inFlight.Add(-1)
		time.Sleep(20 * time.Millisecond)
		w.WriteHeader(http.StatusOK)
	}))

	for _, tc := range []struct {
		name   string
		method string
		path   string
		body   string
	}{
		{"models listing", http.MethodGet, "/v1/models", ""},
		// /v1/responses states its prompt in `input`, which this package
		// does not model; unmetered is the honest answer, not a guess.
		{"responses", http.MethodPost, "/v1/responses", `{"input":"hi"}`},
		{"empty object", http.MethodPost, "/v1/chat/completions", `{}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			const callers = 4
			peak.Store(0)
			var wg sync.WaitGroup
			for range callers {
				wg.Add(1)
				go func() {
					defer wg.Done()
					var rdr io.Reader
					if tc.body != "" {
						rdr = strings.NewReader(tc.body)
					}
					req := httptest.NewRequest(tc.method, tc.path, rdr)
					rec := httptest.NewRecorder()
					h.ServeHTTP(rec, req)
					if rec.Code != http.StatusOK {
						t.Errorf("%s %s status = %d, want 200", tc.method, tc.path, rec.Code)
					}
				}()
			}
			wg.Wait()
			// The pool is exactly one default output reserve, so a single
			// erroneous charge serialises all four.
			if got := peak.Load(); got != callers {
				t.Errorf("peak concurrency = %d, want %d: %s reserved pool it cannot use",
					got, callers, tc.path)
			}
		})
	}
	if got := engine.tokenizeHits.Load(); got != 0 {
		t.Errorf("/tokenize was called %d times for requests with no prompt", got)
	}
}

// TestBodyReachesTheHandlerIntact is the invariant a body-reading middleware
// most easily breaks, and the symptom would be every request failing upstream
// rather than anything this package logs.
func TestBodyReachesTheHandlerIntact(t *testing.T) {
	var seen atomic.Value
	g := New(Options{Capacity: func() (int, bool) { return 131072, true }})
	h := g.Wrap(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, err := readAll(r)
		if err != nil {
			t.Errorf("handler could not read the body: %v", err)
		}
		seen.Store(string(raw))
		if r.ContentLength != int64(len(raw)) {
			t.Errorf("ContentLength = %d, body is %d bytes", r.ContentLength, len(raw))
		}
	}))

	body := chatBody(120)
	postChat(t, h, body)
	if got := seen.Load(); got != body {
		t.Errorf("handler saw a different body than the client sent")
	}
}

// TestCapacityChangeRebuildsTheBudget is the relaunch case: a card edit
// changes -c and the engine comes back with a different pool, so a gate
// holding the old number would enforce it until the process restarted.
func TestCapacityChangeRebuildsTheBudget(t *testing.T) {
	var capacity atomic.Int64
	capacity.Store(4096)
	g := New(Options{Capacity: func() (int, bool) { return int(capacity.Load()), true }})

	_, first := g.semaphore()
	if first != 4096 {
		t.Fatalf("capacity = %d, want 4096", first)
	}
	capacity.Store(32768)
	sem, second := g.semaphore()
	if second != 32768 {
		t.Fatalf("capacity = %d, want the relaunched engine's 32768", second)
	}
	if !sem.TryAcquire(20000) {
		t.Error("the rebuilt budget still refuses a request the new pool holds")
	}
}

// TestConcurrentAdmissionsNeverExceedThePool is the property the whole package
// claims. Asserted by watching the engine rather than the semaphore, because
// the semaphore agreeing with itself proves nothing.
func TestConcurrentAdmissionsNeverExceedThePool(t *testing.T) {
	const (
		pool      = 12288
		perTokens = 3000
		reserve   = 2048
	)
	engine := &fakeEngine{tokensPerCall: perTokens, chatDelay: 20 * time.Millisecond}
	srv := engine.server(t)

	var inFlight, peak atomic.Int32
	tracked := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n := inFlight.Add(1)
		for {
			old := peak.Load()
			if n <= old || peak.CompareAndSwap(old, n) {
				break
			}
		}
		defer inFlight.Add(-1)
		proxyTo(t, srv.URL).ServeHTTP(w, r)
	})

	g := New(Options{
		Capacity:  func() (int, bool) { return pool, true },
		Tokenizer: &Tokenizer{BaseURL: srv.URL},
		Wait:      5 * time.Second,
	})
	h := g.Wrap(tracked)

	var wg sync.WaitGroup
	for range 12 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if rec := postChat(t, h, chatBody(4000)); rec.Code != http.StatusOK {
				t.Errorf("status = %d, want 200 within a 5s wait", rec.Code)
			}
		}()
	}
	wg.Wait()

	want := int32(pool / (perTokens + reserve))
	if got := peak.Load(); got > want {
		t.Errorf("peak concurrency at the engine = %d, want at most %d (%d-token pool, %d per request)",
			got, want, pool, perTokens+reserve)
	}
	if got := peak.Load(); got < 2 {
		t.Errorf("peak concurrency = %d; the gate serialised traffic the pool could have overlapped", got)
	}
}

func waitFor(t *testing.T, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(2 * time.Millisecond)
	}
	t.Fatal("timed out waiting for the condition")
}

// proxyTo is the handler the gate actually fronts in production: a streaming
// reverse proxy. Tests go through it rather than a bare handler because the
// release contract depends on when ReverseProxy returns, and a stub that
// returns immediately would not exercise that at all.
func proxyTo(t *testing.T, target string) http.Handler {
	t.Helper()
	u, err := url.Parse(target)
	if err != nil {
		t.Fatalf("parse engine URL: %v", err)
	}
	rp := httputil.NewSingleHostReverseProxy(u)
	rp.FlushInterval = -1
	return rp
}

func readAll(r *http.Request) ([]byte, error) {
	if r.Body == nil {
		return nil, nil
	}
	defer func() { _ = r.Body.Close() }()
	return io.ReadAll(r.Body)
}
