package main

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/llm-init/llm-init/internal/comfyconsole"
	"github.com/llm-init/llm-init/internal/config"
	"github.com/llm-init/llm-init/internal/version"
)

// runComfyUI is `llm-init comfyui`: the Model Console of a single-workflow
// ComfyUI app (see internal/comfyconsole). It reads its own small env set
// and never loads the LLM config.
//
//	COMFY_APP_CONFIG  app config JSON (default /etc/comfy-app/app-config.json)
//	COMFY_MODELS_DIR  ComfyUI models root (required)
//	COMFYUI_URL       ComfyUI base URL (default http://127.0.0.1:8188)
//	PORT              listen port (default 8090)
//	HF_TOKEN          optional bearer token for huggingface.co downloads
//	HF_ENDPOINT       optional Hugging Face mirror; huggingface.co model URLs are fetched from it
//	COMFY_DOWNLOAD_CONNECTIONS  concurrent byte ranges per large model (default 8)
//	LOG_LEVEL / LOG_FORMAT  as in the LLM mode
func runComfyUI(getenv config.Getenv, stderr *os.File) int {
	env := func(k, def string) string {
		if v := strings.TrimSpace(getenv(k)); v != "" {
			return v
		}
		return def
	}
	logger := newLogger(env("LOG_LEVEL", "info"), env("LOG_FORMAT", config.LogFormatJSON), stderr)
	slog.SetDefault(logger)

	cfgFile := env("COMFY_APP_CONFIG", "/etc/comfy-app/app-config.json")
	cfg, err := comfyconsole.LoadConfig(cfgFile)
	if err != nil {
		fmt.Fprintf(stderr, "config error: %v\n", err)
		return exitConfig
	}
	mirrored, err := cfg.UseMirror(env("HF_ENDPOINT", ""))
	if err != nil {
		fmt.Fprintf(stderr, "config error: %v\n", err)
		return exitConfig
	}
	modelsDir := env("COMFY_MODELS_DIR", "")
	if modelsDir == "" {
		fmt.Fprintln(stderr, "config error: COMFY_MODELS_DIR is required")
		return exitConfig
	}
	var headers http.Header
	if tok := env("HF_TOKEN", ""); tok != "" {
		headers = http.Header{"Authorization": {"Bearer " + tok}}
	}

	conns := 0
	if v := env("COMFY_DOWNLOAD_CONNECTIONS", ""); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil || n < 1 {
			fmt.Fprintf(stderr, "config error: COMFY_DOWNLOAD_CONNECTIONS must be a positive integer, got %q\n", v)
			return exitConfig
		}
		conns = n
	}

	console := comfyconsole.New(comfyconsole.Options{
		Config:      cfg,
		ModelsDir:   modelsDir,
		ComfyURL:    strings.TrimRight(env("COMFYUI_URL", "http://127.0.0.1:8188"), "/"),
		Headers:     headers,
		Connections: conns,
		Logger:      logger,
	})

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	addr := net.JoinHostPort("", env("PORT", "8090"))
	ln, err := net.Listen("tcp", addr)
	if err != nil {
		fmt.Fprintf(stderr, "listen %s: %v\n", addr, err)
		return exitControlPlane
	}
	srv := &http.Server{Handler: console.Handler(), ReadHeaderTimeout: 10 * time.Second}
	go func() {
		if err := srv.Serve(ln); err != nil && !errors.Is(err, http.ErrServerClosed) {
			logger.Error("console server", "err", err)
			stop()
		}
	}()
	logger.Info("comfyui console started", "version", version.Version, "addr", addr,
		"config", cfgFile, "models_dir", modelsDir, "models", len(cfg.Models), "hf_mirrored", mirrored)

	console.Run(ctx)

	shutdownCtx, cancel := context.WithTimeout(context.Background(), shutdownGracePeriod)
	defer cancel()
	_ = srv.Shutdown(shutdownCtx)
	return exitOK
}
