// Command faultsrv runs the download test suite's fault-injection HTTP
// server as a standalone process so the manual runbook
// (tests/download/README.md) can reproduce, by hand, the exact
// abnormal conditions the automated suite drives.
//
// Example:
//
//	go run ./tests/download/cmd/faultsrv -addr :9000 -size 1048576 \
//	    -support-range -fail-first 2
//
// It prints the listen URL, the payload size, and the payload SHA-256
// (so you can build a `#sha256=` MODEL_SOURCE fragment) and serves
// until interrupted.
package main

import (
	"flag"
	"fmt"
	"log"
	"net/http"
	"time"

	"github.com/llm-init/llm-init/tests/download/faultsrv"
)

func main() {
	addr := flag.String("addr", "127.0.0.1:9000", "listen address")
	size := flag.Int("size", 1<<20, "payload size in bytes")
	supportRange := flag.Bool("support-range", true, "honour Range requests (206 resume)")
	omitLen := flag.Bool("omit-content-length", false, "drop Content-Length header")
	headStatus := flag.Int("head-status", 0, "HEAD status override (e.g. 403)")
	forceStatus := flag.Int("force-status", 0, "GET status override on every request (e.g. 403, 404, 429)")
	failFirst := flag.Int("fail-first", 0, "return 503 for the first N GET attempts")
	failStatus := flag.Int("fail-status", 503, "status used for -fail-first attempts")
	dropAfter := flag.Int("drop-after", 0, "write N bytes then drop the connection")
	dropFirst := flag.Int("drop-first", 0, "limit -drop-after to the first N attempts (0 = all)")
	etagRotate := flag.Int("etag-rotate-after", 0, "rotate the ETag after N GET attempts")
	throttle := flag.Duration("throttle", 0, "sleep between 32 KiB chunks (e.g. 50ms)")
	flag.Parse()

	body := faultsrv.GenerateBody(*size)
	cfg := faultsrv.Config{
		Body:              body,
		SupportRange:      *supportRange,
		OmitContentLength: *omitLen,
		HeadStatus:        *headStatus,
		ForceStatus:       *forceStatus,
		FailFirstN:        int32(*failFirst),
		FailStatus:        *failStatus,
		DropAfter:         *dropAfter,
		DropFirstN:        int32(*dropFirst),
		ETagRotateAfter:   int32(*etagRotate),
		Throttle:          *throttle,
	}

	fmt.Printf("faultsrv listening on http://%s/model.bin\n", *addr)
	fmt.Printf("  payload bytes : %d\n", len(body))
	fmt.Printf("  payload sha256: %s\n", faultsrv.SHA256Hex(body))

	server := &http.Server{
		Addr:              *addr,
		Handler:           faultsrv.Handler(cfg),
		ReadHeaderTimeout: 10 * time.Second,
	}
	log.Fatal(server.ListenAndServe())
}
