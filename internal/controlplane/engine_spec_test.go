package controlplane

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/llm-init/llm-init/internal/config"
	"github.com/llm-init/llm-init/internal/obs"
)

func catalogWithEngineSpec(
	t *testing.T,
	kind config.EngineKind,
	payload string,
	alive bool,
	requests *atomic.Int32,
) EndpointList {
	t.Helper()
	engine := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests.Add(1)
		if r.URL.Path != pathEngineSpec {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		_, _ = w.Write([]byte(payload))
	}))
	t.Cleanup(engine.Close)

	s := endpointsFixture(t, kind, func(o *Options) {
		o.DataPlane = func(_ *http.ServeMux) {}
		withCfg(o, func(c *config.Config) { c.Engine.URL = engine.URL })
		o.Readiness = func() Readiness { return Readiness{EngineAlive: alive} }
	})
	r := get(t, s, "/api/endpoints")
	if r.Status != http.StatusOK {
		t.Fatalf("status = %d, body=%s", r.Status, r.Body)
	}
	return parseEndpoints(t, r.Body)
}

func hasEndpoint(list EndpointList, method, path string) bool {
	for _, e := range list.Endpoints {
		if e.Method == method && e.Path == path {
			return true
		}
	}
	return false
}

func TestEngineSpec_AudioV1IsAuthoritative(t *testing.T) {
	t.Parallel()
	var requests atomic.Int32
	list := catalogWithEngineSpec(t, config.EngineAudio, `{
		"schema_version": 1,
		"base": "qwen",
		"model": "demo-7b",
		"implements": ["stt", "stt_stream", "align"],
		"declares": ["stt", "stt_stream"],
		"serves": ["stt", "stt_stream"],
		"endpoints": [
			{"method":"GET","path":"/v1/models","available":true},
			{"capability":"stt","method":"POST","path":"/v1/audio/transcriptions","description":"Offline transcription","available":true,"async_supported":true,"max_input_seconds":540},
			{"capability":"stt_stream","method":"WS","path":"/v1/audio/stream","description":"Streaming ASR","available":true,"async_supported":false},
			{"capability":"align","method":"POST","path":"/v1/audio/align","available":false,"reason":"not declared in MODEL_SUPPORTS"},
			{"method":"GET","path":"/v1/tasks","available":true},
			{"method":"GET","path":"/v1/tasks/{id}","available":true},
			{"method":"GET","path":"/v1/tasks/{id}/result","available":true},
			{"method":"DELETE","path":"/v1/tasks/{id}","available":true},
			{"method":"GET","path":"/v1/audio/tasks","available":true,"deprecated":true},
			{"method":"GET","path":"/v1/audio/tasks/{id}","available":true,"deprecated":true},
			{"method":"GET","path":"/v1/audio/tasks/{id}/result","available":true,"deprecated":true},
			{"method":"DELETE","path":"/v1/audio/tasks/{id}","available":true,"deprecated":true},
			{"capability":"bad","method":"","path":"/v1/bad","available":true},
			{"capability":"bad-type","method":42,"path":"/v1/bad-type","available":true}
		]
	}`, true, &requests)

	if !hasEndpoint(list, mPOST, "/v1/audio/transcriptions") {
		t.Fatal("audio endpoint was not relayed")
	}
	transcription := findByPath(t, list, mPOST, "/v1/audio/transcriptions")
	if transcription.AsyncSupported == nil || !*transcription.AsyncSupported {
		t.Fatalf("async_supported = %v, want true", transcription.AsyncSupported)
	}
	encoded, err := json.Marshal(list)
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(encoded, []byte("max_input_seconds")) {
		t.Fatalf("engine-only max_input_seconds was relayed: %s", encoded)
	}
	stream := findByPath(t, list, mWS, "/v1/audio/stream")
	if stream.AsyncSupported == nil || *stream.AsyncSupported {
		t.Fatalf("stream async_supported = %v, want false", stream.AsyncSupported)
	}
	if !hasEndpoint(list, mGET, pathTask) || !hasEndpoint(list, mDELETE, pathTask) {
		t.Fatal("GET and DELETE task rows must be distinguished by method")
	}
	if !hasEndpoint(list, "WS", "/v1/audio/stream") {
		t.Fatal("audio WebSocket row was not relayed")
	}
	legacy := findByPath(t, list, mGET, "/v1/audio/tasks/{id}")
	if !strings.Contains(legacy.Description, "deprecated") {
		t.Fatalf("legacy row description = %q, want deprecated marker", legacy.Description)
	}
	if hasEndpoint(list, mPOST, pathChatCompletions) {
		t.Fatal("valid v1 report must drop undeclared proxied rows")
	}
	if hasEndpoint(list, "", "/v1/bad") {
		t.Fatal("malformed endpoint row must be ignored")
	}
}

func TestEngineSpec_AudioV2RelaysOperationContractAndUnknownExtensions(t *testing.T) {
	t.Parallel()
	var requests atomic.Int32
	list := catalogWithEngineSpec(t, config.EngineAudio, `{
		"schema_version":2,"model":"demo-7b","implements":["tts"],"declares":["tts"],"serves":["tts"],
		"endpoints":[{"method":"POST","path":"/v1/audio/speech","description":"Speak","available":true,
		"operation_id":"speech.synthesize","protocol":"openai.audio.v1","transport":"http","sync_supported":true,
		"async_supported":true,"input_modalities":["text"],"output_modalities":["audio"],
		"output_formats":["wav","pcm"],"sample_rates":[24000],
		"parameters":[{"name":"speed","type":"number","default":1,"minimum":0.5,"maximum":2}],
		"limits":{"max_text_characters":4096},"resource_scope":"model","vendor_hint":{"pace":"native"}}]
	}`, true, &requests)
	row := findByPath(t, list, mPOST, "/v1/audio/speech")
	if row.OperationID != "speech.synthesize" || row.Protocol != "openai.audio.v1" || row.Transport != "http" {
		t.Fatalf("operation metadata = %+v", row)
	}
	if row.SyncSupported == nil || !*row.SyncSupported || row.AsyncSupported == nil || !*row.AsyncSupported {
		t.Fatalf("execution modes = sync:%v async:%v", row.SyncSupported, row.AsyncSupported)
	}
	if len(row.InputModalities) != 1 || row.InputModalities[0] != "text" ||
		len(row.OutputModalities) != 1 || row.OutputModalities[0] != "audio" {
		t.Fatalf("modalities = in:%v out:%v", row.InputModalities, row.OutputModalities)
	}
	if len(row.OutputFormats) != 2 || row.OutputFormats[0] != "wav" || row.OutputFormats[1] != "pcm" ||
		len(row.SampleRates) != 1 || row.SampleRates[0] != 24000 ||
		row.Limits["max_text_characters"] == nil || row.ResourceScope != "model" {
		t.Fatalf("constraints = %+v", row)
	}
	if len(row.Parameters) != 1 || row.Parameters[0].Name != "speed" ||
		row.Parameters[0].Minimum == nil || *row.Parameters[0].Minimum != 0.5 ||
		row.Parameters[0].Maximum == nil || *row.Parameters[0].Maximum != 2 {
		t.Fatalf("parameters = %+v", row.Parameters)
	}
	if row.Extensions["vendor_hint"] == nil {
		t.Fatalf("unknown v2 extension was dropped: %+v", row.Extensions)
	}
}

func TestEngineSpec_AudioV2RelaysMultiSpanTranscription(t *testing.T) {
	t.Parallel()
	var requests atomic.Int32
	list := catalogWithEngineSpec(t, config.EngineAudio, `{
		"schema_version":2,"model":"demo-asr","implements":["stt"],"declares":["stt"],"serves":["stt"],
		"endpoints":[{"capability":"stt","method":"POST","path":"/v1/audio/transcriptions",
		"description":"Offline transcription","available":true,"operation_id":"audio.transcribe",
		"protocol":"openai.audio.v1","transport":"http","sync_supported":true,
		"async_supported":true,"required_supports":["stt"],"input_modalities":["audio"],
		"output_modalities":["text"],"parameters":[{"name":"segments","type":"array"}]}]
	}`, true, &requests)
	row := findByPath(t, list, mPOST, "/v1/audio/transcriptions")
	if row.OperationID != "audio.transcribe" || row.Protocol != "openai.audio.v1" {
		t.Fatalf("operation metadata = %+v", row)
	}
	if len(row.Parameters) != 1 || row.Parameters[0].Name != "segments" || row.Parameters[0].Type != "array" {
		t.Fatalf("parameters = %+v, want the engine's segments declaration", row.Parameters)
	}
}

func TestEngineSpec_AuthoritativeV1OverridesStaticAvailabilityOnly(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		name   string
		kind   config.EngineKind
		method string
		path   string
		reason string
	}{
		{
			name:   "audio",
			kind:   config.EngineAudio,
			method: mGET,
			path:   pathTasks,
			reason: "audio task queue is not ready",
		},
		{
			name:   "ocr",
			kind:   config.EngineOCR,
			method: mPOST,
			path:   pathOCR,
			reason: "OCR pipeline is not ready",
		},
	} {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			var requests atomic.Int32
			list := catalogWithEngineSpec(t, tc.kind, `{
				"schema_version":1,
				"model":"demo-7b",
				"implements":[],
				"declares":[],
				"serves":[],
				"endpoints":[{
					"method":"`+tc.method+`",
					"path":"`+tc.path+`",
					"description":"engine description must not replace local metadata",
					"available":false,
					"reason":"`+tc.reason+`"
				}]
			}`, true, &requests)

			row := findByPath(t, list, tc.method, tc.path)
			if row.Available {
				t.Fatalf("%s %s available=true, want engine-reported false", tc.method, tc.path)
			}
			if len(row.Reasons) != 1 || row.Reasons[0] != tc.reason {
				t.Fatalf("%s %s reasons=%v, want %q", tc.method, tc.path, row.Reasons, tc.reason)
			}
			if row.Category != categoryOpenAI || row.Group == groupEngine ||
				row.Description == "engine description must not replace local metadata" ||
				row.CurlHint == "" {
				t.Fatalf("%s %s metadata was not preserved: %+v", tc.method, tc.path, row)
			}
		})
	}
}

func TestEngineSpec_AuthoritativeV1CannotOverrideOwnedRows(t *testing.T) {
	t.Parallel()
	var requests atomic.Int32
	list := catalogWithEngineSpec(t, config.EngineAudio, `{
		"schema_version":1,
		"model":"demo-7b",
		"implements":[],
		"declares":[],
		"serves":[],
		"endpoints":[
			{"method":"GET","path":"/v1/models","available":true},
			{"method":"GET","path":"/healthz","available":false,"reason":"engine health failed"},
			{"method":"GET","path":"/metrics","available":false,"reason":"engine metrics failed"},
			{"method":"GET","path":"/api/progress","available":false,"reason":"engine progress failed"},
			{"method":"GET","path":"/","available":false,"reason":"engine UI failed"}
		]
	}`, true, &requests)

	for _, row := range []struct {
		method string
		path   string
	}{
		{mGET, "/healthz"},
		{mGET, "/metrics"},
		{mGET, "/api/progress"},
		{mGET, "/"},
	} {
		got := findByPath(t, list, row.method, row.path)
		if !got.Available || len(got.Reasons) != 0 || got.Group == groupEngine {
			t.Errorf("%s %s was changed by engine report: %+v", row.method, row.path, got)
		}
	}
}

func TestEngineSpec_OCRV1DoesNotRequireBase(t *testing.T) {
	t.Parallel()
	var requests atomic.Int32
	list := catalogWithEngineSpec(t, config.EngineOCR, `{
		"schema_version": 1,
		"model": "demo-7b",
		"implements": ["ocr"],
		"declares": ["ocr"],
		"serves": ["ocr"],
		"endpoints": [
			{"capability":"ocr","method":"GET","path":"/v1/models","available":true},
			{"capability":"ocr","method":"POST","path":"/v1/ocr","available":true},
			{"capability":"ocr","method":"GET","path":"/v1/tasks","available":true},
			{"capability":"ocr","method":"GET","path":"/v1/tasks/{id}","available":true},
			{"capability":"ocr","method":"GET","path":"/v1/tasks/{id}/result","available":true},
			{"capability":"ocr","method":"DELETE","path":"/v1/tasks/{id}","available":true},
			{"capability":"ocr","method":"GET","path":"/v1/ocr/queue","available":true,"deprecated":true},
			{"capability":"ocr","method":"GET","path":"/v1/ocr/jobs/{id}","available":true,"deprecated":true},
			{"capability":"ocr","method":"DELETE","path":"/v1/ocr/jobs/{id}","available":true,"deprecated":true}
		]
	}`, true, &requests)

	if !hasEndpoint(list, mPOST, pathOCR) || !hasEndpoint(list, mGET, pathTasks) {
		t.Fatal("OCR v1 report without base was not relayed")
	}
	if hasEndpoint(list, mPOST, pathChatCompletions) {
		t.Fatal("valid OCR v1 report must be authoritative")
	}
}

func TestEngineSpec_UnknownAndLegacyVersionsRelayWithoutAuthority(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		name    string
		version string
	}{
		{name: "unknown", version: `"schema_version": 3,`},
		{name: "legacy", version: ""},
	} {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			var requests atomic.Int32
			list := catalogWithEngineSpec(t, config.EngineAudio, `{
				`+tc.version+`
				"model": "other-model",
				"endpoints": [
					{"capability":"stt","method":"POST","path":"/v1/audio/transcriptions","available":true},
					{"method":"GET","path":"/v1/models","available":false,"reason":"must not override static row"}
				]
			}`, true, &requests)
			if !hasEndpoint(list, mPOST, "/v1/audio/transcriptions") {
				t.Fatal("formatted endpoint row was not relayed")
			}
			legacy := findByPath(t, list, mPOST, "/v1/audio/transcriptions")
			if legacy.AsyncSupported != nil {
				t.Fatalf("absent optional fields became values: %+v", legacy)
			}
			if !hasEndpoint(list, mPOST, pathChatCompletions) {
				t.Fatal("non-v1 report must not drop llm-init proxied rows")
			}
			models := findByPath(t, list, mGET, pathModels)
			diagnostic := strings.Join(models.Reasons, " ")
			if !models.Available ||
				!strings.Contains(diagnostic, "other-model") ||
				!strings.Contains(diagnostic, "demo-7b") ||
				strings.Contains(diagnostic, "must not override") ||
				models.Group == groupEngine ||
				models.Description == "" {
				t.Fatalf("non-v1 report changed static row: %+v", models)
			}
		})
	}
}

func TestEngineSpec_InvalidV1IsIgnored(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		name    string
		payload string
	}{
		{
			name: "missing model",
			payload: `{"schema_version":1,"implements":[],"declares":[],"serves":[],
				"endpoints":[{"method":"POST","path":"/v1/audio/transcriptions","available":true}]}`,
		},
		{
			name: "missing implements",
			payload: `{"schema_version":1,"model":"demo-7b","declares":[],"serves":[],
				"endpoints":[{"method":"POST","path":"/v1/audio/transcriptions","available":true}]}`,
		},
		{
			name:    "missing endpoints",
			payload: `{"schema_version":1,"model":"demo-7b","implements":[],"declares":[],"serves":[]}`,
		},
		{
			name:    "empty endpoints",
			payload: `{"schema_version":1,"model":"demo-7b","implements":[],"declares":[],"serves":[],"endpoints":[]}`,
		},
		{
			name:    "nonempty endpoints with no usable row",
			payload: `{"schema_version":1,"model":"demo-7b","implements":[],"declares":[],"serves":[],"endpoints":[{}]}`,
		},
		{
			name: "all endpoint rows have wrong types",
			payload: `{"schema_version":1,"model":"demo-7b","implements":[],"declares":[],"serves":[],
				"endpoints":[{"method":42,"path":true},{"method":[],"path":{}}]}`,
		},
		{
			name: "missing available",
			payload: `{"schema_version":1,"model":"demo-7b","implements":[],"declares":[],"serves":[],
				"endpoints":[{"method":"POST","path":"/v1/audio/transcriptions"}]}`,
		},
		{
			name: "available has wrong type",
			payload: `{"schema_version":1,"model":"demo-7b","implements":[],"declares":[],"serves":[],
				"endpoints":[{"method":"POST","path":"/v1/audio/transcriptions","available":"yes"}]}`,
		},
		{
			name: "method has whitespace",
			payload: `{"schema_version":1,"model":"demo-7b","implements":[],"declares":[],"serves":[],
				"endpoints":[{"method":" POST","path":"/v1/audio/transcriptions","available":true}]}`,
		},
		{
			name: "path has whitespace",
			payload: `{"schema_version":1,"model":"demo-7b","implements":[],"declares":[],"serves":[],
				"endpoints":[{"method":"POST","path":"/v1/audio/transcriptions ","available":true}]}`,
		},
		{
			name: "relative path",
			payload: `{"schema_version":1,"model":"demo-7b","implements":[],"declares":[],"serves":[],
				"endpoints":[{"method":"POST","path":"v1/audio/transcriptions","available":true}]}`,
		},
		{
			name: "method contains internal whitespace",
			payload: `{"schema_version":1,"model":"demo-7b","implements":[],"declares":[],"serves":[],
				"endpoints":[{"method":"G ET","path":"/v1/audio/transcriptions","available":true}]}`,
		},
		{
			name: "method has suffix token",
			payload: `{"schema_version":1,"model":"demo-7b","implements":[],"declares":[],"serves":[],
				"endpoints":[{"method":"POST X","path":"/v1/audio/transcriptions","available":true}]}`,
		},
		{
			name: "unknown method",
			payload: `{"schema_version":1,"model":"demo-7b","implements":[],"declares":[],"serves":[],
				"endpoints":[{"method":"BREW","path":"/v1/audio/transcriptions","available":true}]}`,
		},
		{
			name: "path contains whitespace",
			payload: `{"schema_version":1,"model":"demo-7b","implements":[],"declares":[],"serves":[],
				"endpoints":[{"method":"POST","path":"/bad path","available":true}]}`,
		},
		{
			name: "path contains query",
			payload: `{"schema_version":1,"model":"demo-7b","implements":[],"declares":[],"serves":[],
				"endpoints":[{"method":"POST","path":"/x?y=1","available":true}]}`,
		},
		{
			name: "path contains fragment",
			payload: `{"schema_version":1,"model":"demo-7b","implements":[],"declares":[],"serves":[],
				"endpoints":[{"method":"POST","path":"/x#frag","available":true}]}`,
		},
		{
			name: "all rows invalid",
			payload: `{"schema_version":1,"model":"demo-7b","implements":[],"declares":[],"serves":[],
				"endpoints":[
					{"method":"POST","path":"/missing-available"},
					{"method":" POST","path":"/whitespace-method","available":true},
					{"method":"POST","path":"relative","available":true}
				]}`,
		},
		{
			name: "wrong structural type",
			payload: `{"schema_version":1,"model":"demo-7b","implements":"stt","declares":[],"serves":[],
				"endpoints":[{"method":"POST","path":"/v1/audio/transcriptions","available":true}]}`,
		},
	} {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			var requests atomic.Int32
			list := catalogWithEngineSpec(t, config.EngineAudio, tc.payload, true, &requests)
			if hasEndpoint(list, mPOST, "/v1/audio/transcriptions") {
				t.Fatal("invalid v1 report must not be relayed")
			}
			if !hasEndpoint(list, mPOST, pathChatCompletions) {
				t.Fatal("invalid v1 report must not drop llm-init proxied rows")
			}
		})
	}
}

func TestEngineSpec_AcceptsContractMethodTokensAndTemplates(t *testing.T) {
	t.Parallel()
	var requests atomic.Int32
	list := catalogWithEngineSpec(t, config.EngineAudio, `{
		"schema_version":1,
		"model":"demo-7b",
		"implements":[],
		"declares":[],
		"serves":[],
		"endpoints":[
			{"method":"GET","path":"/contract/{id}/get","available":true},
			{"method":"POST","path":"/contract/{id}/post","available":true},
			{"method":"DELETE","path":"/contract/{id}/delete","available":true},
			{"method":"WS","path":"/contract/{id}/ws","available":true},
			{"method":"PUT","path":"/contract/{id}/put","available":true},
			{"method":"PATCH","path":"/contract/{id}/patch","available":true},
			{"method":"HEAD","path":"/contract/{id}/head","available":true},
			{"method":"OPTIONS","path":"/contract/{id}/options","available":true}
		]
	}`, true, &requests)

	for _, method := range []string{"GET", "POST", "DELETE", "WS", "PUT", "PATCH", "HEAD", "OPTIONS"} {
		if !hasEndpoint(list, method, "/contract/{id}/"+strings.ToLower(method)) {
			t.Errorf("%s contract row was not relayed", method)
		}
	}
}

func TestEngineSpec_V1IgnoresBadRowsButRemainsAuthoritative(t *testing.T) {
	t.Parallel()
	var requests atomic.Int32
	list := catalogWithEngineSpec(t, config.EngineAudio, `{
		"schema_version":1,
		"model":"demo-7b",
		"implements":[],
		"declares":[],
		"serves":[],
		"endpoints":[
			{"method":"GET","path":"/v1/models","available":true},
			{"method":"POST","path":"/missing-available"},
			{"method":" POST","path":"/whitespace-method","available":true},
			{"method":"POST","path":"relative","available":true}
		]
	}`, true, &requests)

	if !hasEndpoint(list, mGET, pathModels) {
		t.Fatal("valid row was not retained")
	}
	for _, path := range []string{"/missing-available", "/whitespace-method", "relative"} {
		for _, row := range list.Endpoints {
			if row.Path == path {
				t.Errorf("invalid row was relayed: %+v", row)
			}
		}
	}
	if hasEndpoint(list, mPOST, pathChatCompletions) {
		t.Fatal("v1 with one valid row must remain authoritative")
	}
}

func TestEngineSpec_ModelMismatchIsDiagnosticOnly(t *testing.T) {
	t.Parallel()
	var requests atomic.Int32
	list := catalogWithEngineSpec(t, config.EngineAudio, `{
		"schema_version":1,
		"model":"other-model",
		"implements":["stt"],
		"declares":["stt"],
		"serves":["stt"],
		"endpoints":[
			{"capability":"stt","method":"GET","path":"/v1/models","available":true}
		]
	}`, true, &requests)

	row := findByPath(t, list, mGET, pathModels)
	if !row.Available {
		t.Fatalf("model mismatch must not mark endpoint unavailable: %+v", row)
	}
	diagnostic := strings.Join(row.Reasons, " ")
	if !strings.Contains(diagnostic, "other-model") || !strings.Contains(diagnostic, "demo-7b") {
		t.Fatalf("reasons = %v, want engine and configured model names", row.Reasons)
	}
}

func TestEngineSpec_OCRModelMismatchSurvivesCatalogFilter(t *testing.T) {
	t.Parallel()
	var requests atomic.Int32
	list := catalogWithEngineSpec(t, config.EngineOCR, `{
		"schema_version":1,
		"model":"other-ocr-model",
		"implements":["ocr"],
		"declares":["ocr"],
		"serves":["ocr"],
		"endpoints":[
			{"capability":"ocr","method":"GET","path":"/v1/models","available":true},
			{"capability":"ocr","method":"POST","path":"/v1/ocr","available":true}
		]
	}`, true, &requests)

	row := findByPath(t, list, mPOST, pathOCR)
	if !row.Available {
		t.Fatalf("OCR model mismatch must not mark endpoint unavailable: %+v", row)
	}
	diagnostic := strings.Join(row.Reasons, " ")
	if !strings.Contains(diagnostic, "other-ocr-model") || !strings.Contains(diagnostic, "demo-7b") {
		t.Fatalf("reasons = %v, want engine and configured model names", row.Reasons)
	}
}

func TestApplyDataPlaneCatalog_ReasonPrecedence(t *testing.T) {
	t.Parallel()
	const diagnostic = "engine model mismatch"
	t.Run("allowed preserves diagnostic", func(t *testing.T) {
		available, reasons := applyDataPlaneCatalog(
			config.ModelOCR, true, mPOST, pathOCR, true, []string{diagnostic})
		if !available || len(reasons) != 1 || reasons[0] != diagnostic {
			t.Fatalf("available=%v reasons=%v", available, reasons)
		}
	})
	t.Run("disallowed uses mode reason only", func(t *testing.T) {
		available, reasons := applyDataPlaneCatalog(
			config.ModelOCR, true, mPOST, pathChatCompletions, true, []string{diagnostic})
		if available || len(reasons) != 1 ||
			!strings.Contains(reasons[0], "MODEL_MODE=ocr") ||
			strings.Contains(reasons[0], diagnostic) {
			t.Fatalf("available=%v reasons=%v", available, reasons)
		}
	})
	t.Run("registrar off uses wiring reason only", func(t *testing.T) {
		available, reasons := applyDataPlaneCatalog(
			config.ModelOCR, false, mPOST, pathOCR, true, []string{diagnostic})
		if available || len(reasons) != 1 || reasons[0] != dataPlaneRegistrarOff {
			t.Fatalf("available=%v reasons=%v", available, reasons)
		}
	})
}

func TestEngineSpec_DeadEngineKeepsStaticTaskFallback(t *testing.T) {
	t.Parallel()
	for _, kind := range []config.EngineKind{config.EngineAudio, config.EngineOCR} {
		kind := kind
		t.Run(string(kind), func(t *testing.T) {
			t.Parallel()
			var requests atomic.Int32
			list := catalogWithEngineSpec(t, kind, `{}`, false, &requests)
			for _, row := range []struct {
				method string
				path   string
			}{
				{mGET, pathTasks},
				{mGET, pathTask},
				{mGET, pathTaskResult},
				{mDELETE, pathTask},
			} {
				assertAvail(t, list, row.method, row.path, true)
			}
			if requests.Load() != 0 {
				t.Fatalf("dead engine received %d engine-spec requests", requests.Load())
			}
		})
	}
}

// engineSpecMetrics drives one catalog read against payload and returns the
// metrics the relay recorded while doing it.
func engineSpecMetrics(
	t *testing.T,
	payload string,
	status int,
) *obs.Metrics {
	t.Helper()
	m := obs.NewMetrics()
	engine := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != pathEngineSpec {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		w.WriteHeader(status)
		_, _ = w.Write([]byte(payload))
	}))
	t.Cleanup(engine.Close)

	s := endpointsFixture(t, config.EngineAudio, func(o *Options) {
		o.DataPlane = func(_ *http.ServeMux) {}
		o.Metrics = m
		withCfg(o, func(c *config.Config) { c.Engine.URL = engine.URL })
		o.Readiness = func() Readiness { return Readiness{EngineAlive: true} }
	})
	if r := get(t, s, "/api/endpoints"); r.Status != http.StatusOK {
		t.Fatalf("status = %d, body=%s", r.Status, r.Body)
	}
	return m
}

// counterValue reads one labelled counter sample, or -1 when the series was
// never touched.
func counterValue(t *testing.T, m *obs.Metrics, name string, labels map[string]string) float64 {
	t.Helper()
	families, err := m.Registry().Gather()
	if err != nil {
		t.Fatal(err)
	}
	for _, family := range families {
		if family.GetName() != name {
			continue
		}
	sample:
		for _, sample := range family.GetMetric() {
			for key, want := range labels {
				found := false
				for _, label := range sample.GetLabel() {
					if label.GetName() == key && label.GetValue() == want {
						found = true
						break
					}
				}
				if !found {
					continue sample
				}
			}
			if c := sample.GetCounter(); c != nil {
				return c.GetValue()
			}
		}
	}
	return -1
}

// TestEngineSpec_RefusedRowsAreCounted covers the reason a capability can
// disappear without a trace: a v2 row missing one contract field is dropped,
// the catalog then looks exactly as if the engine never declared it, and the
// gateway downstream answers "operation not supported" for a route the
// engine really serves. The counter is what names the missing field.
func TestEngineSpec_RefusedRowsAreCounted(t *testing.T) {
	t.Parallel()
	m := engineSpecMetrics(t, `{
		"schema_version":2,
		"model":"demo-7b",
		"implements":["stt"],
		"declares":["stt"],
		"serves":["stt"],
		"endpoints":[
			{"method":"POST","path":"/v1/audio/transcriptions","available":true,
			 "operation_id":"audio.transcribe","protocol":"openai.audio.v1",
			 "transport":"http","sync_supported":true},
			{"method":"POST","path":"/v1/audio/vad","available":true,
			 "operation_id":"audio.vad","protocol":"openai.audio.v1","transport":"http"},
			{"method":"POST","path":"/v1/audio/enhance","available":true,
			 "protocol":"openai.audio.v1","transport":"http","sync_supported":true},
			{"method":"POST","path":"/v1/audio/align","available":true,
			 "operation_id":"audio.align","protocol":"openai.audio.v1",
			 "transport":"carrier-pigeon","sync_supported":true}
		]
	}`, http.StatusOK)

	for reason, want := range map[string]float64{
		dropMissingSyncSupported: 1,
		dropMissingOperationID:   1,
		dropBadTransport:         1,
	} {
		if got := counterValue(t, m, "llm_init_engine_spec_rows_dropped_total",
			map[string]string{"reason": reason}); got != want {
			t.Errorf("dropped[%s] = %v want %v", reason, got, want)
		}
	}
	if got := counterValue(t, m, "llm_init_engine_spec_fetch_total",
		map[string]string{"result": specFetchOK}); got != 1 {
		t.Errorf("fetch[ok] = %v want 1", got)
	}
}

// TestEngineSpec_UnreachableEngineIsDistinguishableFromSilence is the whole
// point of the fetch counter: both cases serve an empty report, and without
// the label an operator cannot tell "this engine declares nothing" from "we
// could not ask it".
func TestEngineSpec_UnreachableEngineIsDistinguishableFromSilence(t *testing.T) {
	t.Parallel()
	m := engineSpecMetrics(t, `{}`, http.StatusInternalServerError)
	if got := counterValue(t, m, "llm_init_engine_spec_fetch_total",
		map[string]string{"result": specFetchHTTPStatus}); got != 1 {
		t.Errorf("fetch[http_status] = %v want 1", got)
	}
	if got := counterValue(t, m, "llm_init_engine_spec_fetch_total",
		map[string]string{"result": specFetchOK}); got != -1 {
		t.Errorf("a 500 must not count as ok, got %v", got)
	}
}

// engineSpecStateFixture drives engineEndpoints against a handler the test
// controls, with a clock it can move past the cache TTL.
func engineSpecStateFixture(
	t *testing.T,
	handler http.HandlerFunc,
) (*Server, *time.Time) {
	t.Helper()
	engine := httptest.NewServer(handler)
	t.Cleanup(engine.Close)
	now := time.Now()
	s := endpointsFixture(t, config.EngineAudio, func(o *Options) {
		o.DataPlane = func(_ *http.ServeMux) {}
		withCfg(o, func(c *config.Config) { c.Engine.URL = engine.URL })
		o.Readiness = func() Readiness { return Readiness{EngineAlive: true} }
		o.NowFunc = func() time.Time { return now }
	})
	return s, &now
}

const audioSpecPayload = `{"schema_version":1,"base":"qwen3-asr","model":"demo-7b",
	"implements":["stt"],"declares":["stt"],"serves":["stt"],"endpoints":[
	{"capability":"stt","method":"POST","path":"/v1/audio/transcriptions",
	 "description":"Offline transcription","available":true}]}`

// TestEngineSpec_AnEngineWithoutTheRouteIsNotPending: an engine written
// before this contract answers 404 here and will do so forever. Treating
// that as "no answer yet" would withhold its data plane for the life of the
// process, so the relay has to read a 404 as the settled answer it is.
func TestEngineSpec_AnEngineWithoutTheRouteIsNotPending(t *testing.T) {
	t.Parallel()
	s, _ := engineSpecStateFixture(t, func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusNotFound)
	})
	if got := s.engineEndpoints().state; got != engineSpecSilent {
		t.Fatalf("state = %v, want silent", got)
	}
	list := parseEndpoints(t, get(t, s, "/api/endpoints").Body)
	if e := findByPath(t, list, mPOST, pathChatCompletions); !e.Available {
		t.Errorf("a legacy engine's static rows were withheld (reasons=%v)", e.Reasons)
	}
}

// TestEngineSpec_AnEngineThatFailedToAnswerIsPending: a 5xx is the engine
// failing to answer rather than answering, and reading it as "declares
// nothing" would put the whole static surface back — the audio application
// would advertise chat again, one failing poll at a time.
func TestEngineSpec_AnEngineThatFailedToAnswerIsPending(t *testing.T) {
	t.Parallel()
	s, _ := engineSpecStateFixture(t, func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	})
	if got := s.engineEndpoints().state; got != engineSpecUnknown {
		t.Fatalf("state = %v, want unknown", got)
	}
	list := parseEndpoints(t, get(t, s, "/api/endpoints").Body)
	if e := findByPath(t, list, mPOST, pathChatCompletions); e.Available {
		t.Error("an audio application advertised chat while its engine was failing to answer")
	}
}

// TestEngineSpec_ABlipDoesNotReopenWhatADeclarationClosed: once an engine
// has declared its routes, a fetch that fails afterwards has not withdrawn
// them. Replacing the report with an empty one would re-open every proxied
// row the declaration had removed, and it is the poll after the failure —
// not the failure itself — that a dashboard or a model sync catches.
func TestEngineSpec_ABlipDoesNotReopenWhatADeclarationClosed(t *testing.T) {
	t.Parallel()
	var broken atomic.Bool
	s, now := engineSpecStateFixture(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != pathEngineSpec {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		if broken.Load() {
			w.WriteHeader(http.StatusInternalServerError)
			return
		}
		_, _ = w.Write([]byte(audioSpecPayload))
	})
	first := s.engineEndpoints()
	if first.state != engineSpecDeclared || len(first.rows) == 0 {
		t.Fatalf("state = %v, rows = %d", first.state, len(first.rows))
	}

	broken.Store(true)
	*now = now.Add(2 * engineSpecTTL)

	second := s.engineEndpoints()
	if second.state != engineSpecDeclared {
		t.Fatalf("state = %v after one failed poll, want the declaration to stand", second.state)
	}
	if len(second.rows) != len(first.rows) {
		t.Errorf("rows = %d, want the %d the engine declared", len(second.rows), len(first.rows))
	}
}

// TestEngineSpec_AnEngineThatWentSilentIsBelieved is the other half: a 404
// is a statement about the engine now serving, not a failure to reach one,
// so the declaration it replaces is really gone.
func TestEngineSpec_AnEngineThatWentSilentIsBelieved(t *testing.T) {
	t.Parallel()
	var gone atomic.Bool
	s, now := engineSpecStateFixture(t, func(w http.ResponseWriter, r *http.Request) {
		if gone.Load() || r.URL.Path != pathEngineSpec {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		_, _ = w.Write([]byte(audioSpecPayload))
	})
	if got := s.engineEndpoints().state; got != engineSpecDeclared {
		t.Fatalf("state = %v, want declared", got)
	}

	gone.Store(true)
	*now = now.Add(2 * engineSpecTTL)

	report := s.engineEndpoints()
	if report.state != engineSpecSilent || len(report.rows) != 0 {
		t.Fatalf("state = %v with %d rows, want a silent engine to drop its declaration",
			report.state, len(report.rows))
	}
}

// TestEngineSpec_MalformedVersionedSpecNamesTheMissingFields keeps the
// operator-facing detail honest: refusing a whole spec removes every
// capability at once, so the log has to say which envelope field caused it.
func TestEngineSpec_MalformedVersionedSpecNamesTheMissingFields(t *testing.T) {
	t.Parallel()
	detail := invalidEngineSpecDetail(engineSpec{Model: "  "})
	for _, want := range []string{"model", "implements", "declares", "serves", "endpoints"} {
		if !strings.Contains(detail, want) {
			t.Errorf("detail %q does not name %q", detail, want)
		}
	}
}
