package ollama

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
)

// fakeOllama is a minimal stand-in for the Ollama daemon. Each handler
// hook is set per-test so we can stage failure modes (404 ps, early-EOF
// pull stream, etc.) without inheriting state across tests.
type fakeOllama struct {
	tags      func(w http.ResponseWriter, r *http.Request)
	ps        func(w http.ResponseWriter, r *http.Request)
	pull      func(w http.ResponseWriter, r *http.Request)
	blobs     func(w http.ResponseWriter, r *http.Request)
	create    func(w http.ResponseWriter, r *http.Request)
	deleter   func(w http.ResponseWriter, r *http.Request)
	responses func(w http.ResponseWriter, r *http.Request)
}

func (f *fakeOllama) handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/api/tags", func(w http.ResponseWriter, r *http.Request) {
		if f.tags != nil {
			f.tags(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"models":[]}`))
	})
	mux.HandleFunc("/api/ps", func(w http.ResponseWriter, r *http.Request) {
		if f.ps != nil {
			f.ps(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"models":[]}`))
	})
	mux.HandleFunc("/api/pull", func(w http.ResponseWriter, r *http.Request) {
		if f.pull != nil {
			f.pull(w, r)
			return
		}
		writeNDJSON(w, []map[string]any{{"status": "success"}})
	})
	mux.HandleFunc("/api/blobs/", func(w http.ResponseWriter, r *http.Request) {
		if f.blobs != nil {
			f.blobs(w, r)
			return
		}
		w.WriteHeader(http.StatusNotFound)
	})
	mux.HandleFunc("/api/create", func(w http.ResponseWriter, r *http.Request) {
		if f.create != nil {
			f.create(w, r)
			return
		}
		writeNDJSON(w, []map[string]any{{"status": "success"}})
	})
	mux.HandleFunc("/api/delete", func(w http.ResponseWriter, r *http.Request) {
		if f.deleter != nil {
			f.deleter(w, r)
			return
		}
		w.WriteHeader(http.StatusOK)
	})
	mux.HandleFunc("/v1/responses", func(w http.ResponseWriter, r *http.Request) {
		if f.responses != nil {
			f.responses(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"resp_test","object":"response","status":"completed","output":[]}`))
	})
	return mux
}

// writeNDJSON serialises rows as newline-delimited JSON, the format
// /api/pull and /api/create stream.
func writeNDJSON(w http.ResponseWriter, rows []map[string]any) {
	w.Header().Set("Content-Type", "application/x-ndjson")
	for _, row := range rows {
		b, _ := json.Marshal(row)
		_, _ = w.Write(b)
		_, _ = w.Write([]byte("\n"))
	}
}

func newClient(t *testing.T, fake *fakeOllama) *Client {
	t.Helper()
	srv := httptest.NewServer(fake.handler())
	t.Cleanup(srv.Close)
	return NewClient(srv.URL)
}

func TestClient_TagsAndModelExists(t *testing.T) {
	t.Parallel()
	fake := &fakeOllama{
		tags: func(w http.ResponseWriter, _ *http.Request) {
			_, _ = w.Write([]byte(`{"models":[{"name":"qwen2.5:7b"},{"name":"llama3"}]}`))
		},
	}
	c := newClient(t, fake)

	t.Run("exact match", func(t *testing.T) {
		ok, err := c.ModelExists(context.Background(), "qwen2.5:7b")
		if err != nil || !ok {
			t.Errorf("exact: ok=%v err=%v", ok, err)
		}
	})
	t.Run("prefix tag→base", func(t *testing.T) {
		// looking for "qwen2.5" should match "qwen2.5:7b"
		ok, err := c.ModelExists(context.Background(), "qwen2.5")
		if err != nil || !ok {
			t.Errorf("prefix: ok=%v err=%v", ok, err)
		}
	})
	t.Run("missing", func(t *testing.T) {
		ok, err := c.ModelExists(context.Background(), "mistral:7b")
		if err != nil || ok {
			t.Errorf("missing: ok=%v err=%v", ok, err)
		}
	})
}

func TestClient_TagsBadStatus(t *testing.T) {
	t.Parallel()
	c := newClient(t, &fakeOllama{
		tags: func(w http.ResponseWriter, _ *http.Request) {
			http.Error(w, "boom", http.StatusInternalServerError)
		},
	})
	if _, err := c.Tags(context.Background()); err == nil {
		t.Error("expected error")
	}
}

func TestClient_TagsBadJSON(t *testing.T) {
	t.Parallel()
	c := newClient(t, &fakeOllama{
		tags: func(w http.ResponseWriter, _ *http.Request) {
			_, _ = w.Write([]byte("not json"))
		},
	})
	if _, err := c.Tags(context.Background()); err == nil {
		t.Error("expected decode error")
	}
}

func TestClient_PSOK(t *testing.T) {
	t.Parallel()
	c := newClient(t, &fakeOllama{
		ps: func(w http.ResponseWriter, _ *http.Request) {
			_, _ = w.Write([]byte(`{"models":[{"name":"qwen"}]}`))
		},
	})
	ps, err := c.PS(context.Background())
	if err != nil {
		t.Fatalf("PS: %v", err)
	}
	if len(ps.Models) != 1 || ps.Models[0].Name != "qwen" {
		t.Errorf("ps = %+v", ps)
	}
}

func TestClient_PSNotFound(t *testing.T) {
	t.Parallel()
	c := newClient(t, &fakeOllama{
		ps: func(w http.ResponseWriter, _ *http.Request) {
			http.NotFound(w, nil)
		},
	})
	if _, err := c.PS(context.Background()); err == nil {
		t.Error("expected error on 404")
	}
}

func TestClient_PullSuccess(t *testing.T) {
	t.Parallel()
	frames := []map[string]any{
		{"status": "pulling manifest"},
		{"status": "downloading", "digest": "sha256:abc", "total": 100, "completed": 25},
		{"status": "downloading", "digest": "sha256:abc", "total": 100, "completed": 100},
		{"status": "verifying sha256 digest"},
		{"status": "writing manifest"},
		{"status": "success"},
	}
	c := newClient(t, &fakeOllama{
		pull: func(w http.ResponseWriter, _ *http.Request) {
			writeNDJSON(w, frames)
		},
	})
	var got []PullResponse
	if err := c.Pull(context.Background(), "qwen2.5:7b", func(p PullResponse) {
		got = append(got, p)
	}); err != nil {
		t.Fatalf("Pull: %v", err)
	}
	if len(got) != len(frames) {
		t.Errorf("frames = %d, want %d", len(got), len(frames))
	}
	if got[len(got)-1].Status != "success" {
		t.Errorf("last status = %q", got[len(got)-1].Status)
	}
}

func TestClient_PullEarlyEOF(t *testing.T) {
	t.Parallel()
	c := newClient(t, &fakeOllama{
		pull: func(w http.ResponseWriter, _ *http.Request) {
			writeNDJSON(w, []map[string]any{
				{"status": "downloading", "total": 100, "completed": 50},
			})
			// no "success" frame, just EOF
		},
	})
	err := c.Pull(context.Background(), "x", nil)
	if !errors.Is(err, ErrPullNoSuccess) {
		t.Errorf("err = %v, want ErrPullNoSuccess", err)
	}
}

func TestClient_PullNon200(t *testing.T) {
	t.Parallel()
	c := newClient(t, &fakeOllama{
		pull: func(w http.ResponseWriter, _ *http.Request) {
			http.Error(w, `{"error":"model not found"}`, http.StatusNotFound)
		},
	})
	err := c.Pull(context.Background(), "ghost", nil)
	if err == nil || errors.Is(err, ErrPullNoSuccess) {
		t.Errorf("err = %v, want plain HTTP 404 error", err)
	}
	if !strings.Contains(err.Error(), "404") {
		t.Errorf("error message lost status: %v", err)
	}
}

func TestClient_PullDecodeMidStream(t *testing.T) {
	t.Parallel()
	c := newClient(t, &fakeOllama{
		pull: func(w http.ResponseWriter, _ *http.Request) {
			_, _ = w.Write([]byte(`{"status":"ok"}` + "\n"))
			_, _ = w.Write([]byte("garbage\n"))
		},
	})
	if err := c.Pull(context.Background(), "x", nil); err == nil {
		t.Error("expected decode error mid-stream")
	}
}

func TestClient_BlobExists(t *testing.T) {
	t.Parallel()
	var seen atomic.Int32
	c := newClient(t, &fakeOllama{
		blobs: func(w http.ResponseWriter, r *http.Request) {
			seen.Add(1)
			if strings.HasSuffix(r.URL.Path, "/sha256:present") {
				w.WriteHeader(http.StatusOK)
				return
			}
			w.WriteHeader(http.StatusNotFound)
		},
	})
	ok, err := c.BlobExists(context.Background(), "sha256:present")
	if err != nil || !ok {
		t.Errorf("present: ok=%v err=%v", ok, err)
	}
	ok, err = c.BlobExists(context.Background(), "sha256:absent")
	if err != nil || ok {
		t.Errorf("absent: ok=%v err=%v", ok, err)
	}
	if seen.Load() != 2 {
		t.Errorf("blob hits = %d", seen.Load())
	}
}

func TestClient_BlobExistsBadStatus(t *testing.T) {
	t.Parallel()
	c := newClient(t, &fakeOllama{
		blobs: func(w http.ResponseWriter, _ *http.Request) {
			http.Error(w, "boom", http.StatusInternalServerError)
		},
	})
	if _, err := c.BlobExists(context.Background(), "sha256:x"); err == nil {
		t.Error("expected error on 500")
	}
}

func TestClient_PushBlobSuccess(t *testing.T) {
	t.Parallel()
	body := []byte("hello world from a tiny gguf")
	tmp := filepath.Join(t.TempDir(), "blob.gguf")
	if err := os.WriteFile(tmp, body, 0o644); err != nil {
		t.Fatal(err)
	}

	var got []byte
	c := newClient(t, &fakeOllama{
		blobs: func(w http.ResponseWriter, r *http.Request) {
			if r.Method != http.MethodPost {
				t.Errorf("method = %s", r.Method)
			}
			data, _ := io.ReadAll(r.Body)
			got = data
			w.WriteHeader(http.StatusCreated)
		},
	})
	if err := c.PushBlob(context.Background(), "sha256:x", tmp); err != nil {
		t.Fatalf("PushBlob: %v", err)
	}
	if string(got) != string(body) {
		t.Errorf("body mismatch: got %q want %q", got, body)
	}
}

func TestClient_PushBlobMissingFile(t *testing.T) {
	t.Parallel()
	c := newClient(t, &fakeOllama{})
	err := c.PushBlob(context.Background(), "sha256:x",
		filepath.Join(t.TempDir(), "ghost"))
	if err == nil {
		t.Error("expected error opening missing file")
	}
}

func TestClient_PushBlobBadStatus(t *testing.T) {
	t.Parallel()
	tmp := filepath.Join(t.TempDir(), "blob")
	_ = os.WriteFile(tmp, []byte("x"), 0o644)
	c := newClient(t, &fakeOllama{
		blobs: func(w http.ResponseWriter, _ *http.Request) {
			http.Error(w, "no thanks", http.StatusForbidden)
		},
	})
	if err := c.PushBlob(context.Background(), "sha256:x", tmp); err == nil {
		t.Error("expected error on 403")
	}
}

func TestClient_CreateSuccess(t *testing.T) {
	t.Parallel()
	var gotReq CreateRequest
	c := newClient(t, &fakeOllama{
		create: func(w http.ResponseWriter, r *http.Request) {
			_ = json.NewDecoder(r.Body).Decode(&gotReq)
			writeNDJSON(w, []map[string]any{
				{"status": "parsing"},
				{"status": "writing manifest"},
				{"status": "success"},
			})
		},
	})
	req := CreateRequest{
		Model:    "qwen2.5-7b",
		Files:    map[string]string{"q.gguf": "sha256:abc"},
		Template: "tmpl",
		System:   "sys",
	}
	if err := c.Create(context.Background(), req); err != nil {
		t.Fatalf("Create: %v", err)
	}
	if gotReq.Model != "qwen2.5-7b" || gotReq.Template != "tmpl" {
		t.Errorf("server-side request = %+v", gotReq)
	}
}

func TestClient_CreateNoSuccess(t *testing.T) {
	t.Parallel()
	c := newClient(t, &fakeOllama{
		create: func(w http.ResponseWriter, _ *http.Request) {
			writeNDJSON(w, []map[string]any{{"status": "parsing"}})
		},
	})
	err := c.Create(context.Background(), CreateRequest{Model: "x"})
	if !errors.Is(err, ErrCreateNoSuccess) {
		t.Errorf("err = %v", err)
	}
}

func TestClient_CreateBadStatus(t *testing.T) {
	t.Parallel()
	c := newClient(t, &fakeOllama{
		create: func(w http.ResponseWriter, _ *http.Request) {
			http.Error(w, `{"error":"bad request"}`, http.StatusBadRequest)
		},
	})
	if err := c.Create(context.Background(), CreateRequest{Model: "x"}); err == nil {
		t.Error("expected error")
	}
}

func TestClient_DeleteModel(t *testing.T) {
	t.Parallel()
	var gotBody map[string]string
	c := newClient(t, &fakeOllama{
		deleter: func(w http.ResponseWriter, r *http.Request) {
			_ = json.NewDecoder(r.Body).Decode(&gotBody)
			w.WriteHeader(http.StatusOK)
		},
	})
	if err := c.DeleteModel(context.Background(), "doomed"); err != nil {
		t.Fatalf("DeleteModel: %v", err)
	}
	if gotBody["model"] != "doomed" {
		t.Errorf("body = %v", gotBody)
	}
}

func TestClient_DeleteModelNotFoundOK(t *testing.T) {
	t.Parallel()
	c := newClient(t, &fakeOllama{
		deleter: func(w http.ResponseWriter, _ *http.Request) {
			w.WriteHeader(http.StatusNotFound)
		},
	})
	if err := c.DeleteModel(context.Background(), "x"); err != nil {
		t.Errorf("404 should not error: %v", err)
	}
}

func TestClient_DeleteModelOtherStatus(t *testing.T) {
	t.Parallel()
	c := newClient(t, &fakeOllama{
		deleter: func(w http.ResponseWriter, _ *http.Request) {
			http.Error(w, "no", http.StatusForbidden)
		},
	})
	if err := c.DeleteModel(context.Background(), "x"); err == nil {
		t.Error("expected error")
	}
}

func TestClient_PostJSONStreams(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintf(w, "ack %s", r.URL.Path)
	}))
	defer srv.Close()
	c := NewClient(srv.URL)
	resp, err := c.PostJSON(context.Background(), "/api/echo", []byte(`{"x":1}`))
	if err != nil {
		t.Fatalf("PostJSON: %v", err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	if string(body) != "ack /api/echo" {
		t.Errorf("body = %q", body)
	}
}

func TestClient_PostJSONErrorStatus(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "nope", http.StatusUnauthorized)
	}))
	defer srv.Close()
	c := NewClient(srv.URL)
	// nolint:bodyclose // PostJSON returns (nil, err) on non-2xx status
	// (verified by the resp == nil assertion below); there is no body
	// to close. bodyclose flags this because it cannot reason about
	// the contract.
	resp, err := c.PostJSON(context.Background(), "/api/x", []byte(`{}`))
	if err == nil {
		t.Error("expected error on 401")
	}
	if resp != nil {
		t.Error("resp should be nil on error")
	}
}

func TestClient_BaseURLTrim(t *testing.T) {
	t.Parallel()
	c := NewClient("http://example.com:11434/")
	if c.BaseURL != "http://example.com:11434" {
		t.Errorf("BaseURL = %q", c.BaseURL)
	}
}
