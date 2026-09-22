package translate

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
)

const (
	maxBodyBytes  = 1 << 20 // 1 MiB — matches adapter chat body cap
	maxBatchTexts = 64      // sequential LLM calls; keep bounded
)

// Request/response shapes mirror MTranServer
// (src/controllers/translate.controller.ts + language.controller.ts):
//
//	POST /translate       {from,to,text,html?}        → {result}
//	POST /translate/batch {from,to,texts,html?}       → {results}
//	GET  /languages                                   → {languages,pairs}
//	POST /detect          {text,minConfidence?}       → {language[,confidence]}
//
// html is accepted but ignored (no Bergamot HTML mode on the LLM path).

type translateRequest struct {
	From string `json:"from"`
	To   string `json:"to"`
	Text string `json:"text"`
	HTML *bool  `json:"html,omitempty"`
}

type batchRequest struct {
	From  string   `json:"from"`
	To    string   `json:"to"`
	Texts []string `json:"texts"`
	HTML  *bool    `json:"html,omitempty"`
}

type detectRequest struct {
	Text          string   `json:"text"`
	MinConfidence *float64 `json:"minConfidence,omitempty"`
}

type translateResponse struct {
	Result string `json:"result"`
}

type batchResponse struct {
	Results []string `json:"results"`
}

type languagesResponse struct {
	Languages []string       `json:"languages"`
	Pairs     []languagePair `json:"pairs"`
}

type detectResponse struct {
	Language   string   `json:"language"`
	Confidence *float64 `json:"confidence,omitempty"`
}

func (h *Handler) handleLanguages(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, languagesResponse{
		Languages: h.catalog.Languages,
		Pairs:     h.catalog.Pairs,
	})
}

func (h *Handler) handleDetect(w http.ResponseWriter, r *http.Request) {
	var req detectRequest
	if err := decodeJSONBody(r, &req); err != nil {
		writeRequestErr(w, err)
		return
	}
	got, err := h.DetectLanguage(r.Context(), req.Text)
	if err != nil {
		writeTranslateErr(w, err)
		return
	}

	out := detectResponse{Language: got.Language}
	if req.MinConfidence != nil {
		conf := got.Confidence
		if conf < *req.MinConfidence {
			// MTran clears language when below the threshold.
			out.Language = ""
		}
		out.Confidence = &conf
	}
	declareUsage(w, got.Usage)
	writeJSON(w, http.StatusOK, out)
}

func (h *Handler) handleTranslate(w http.ResponseWriter, r *http.Request) {
	var req translateRequest
	if err := decodeJSONBody(r, &req); err != nil {
		writeRequestErr(w, err)
		return
	}
	result, usage, err := h.translateOne(r, req.From, req.To, req.Text)
	if err != nil {
		writeTranslateErr(w, err)
		return
	}
	declareUsage(w, usage)
	writeJSON(w, http.StatusOK, translateResponse{Result: result})
}

func (h *Handler) handleBatch(w http.ResponseWriter, r *http.Request) {
	var req batchRequest
	if err := decodeJSONBody(r, &req); err != nil {
		writeRequestErr(w, err)
		return
	}
	if req.Texts == nil {
		writeError(w, http.StatusBadRequest, codeInvalidRequest, "texts is required")
		return
	}
	if len(req.Texts) > maxBatchTexts {
		writeError(w, http.StatusBadRequest, codeInvalidRequest,
			fmt.Sprintf("texts exceeds max batch size of %d", maxBatchTexts))
		return
	}
	results := make([]string, 0, len(req.Texts))
	// A batch is one billable request made of N completions, so the row it
	// produces has to carry all N. Declaring only the last would price a
	// 64-text batch as one translation.
	var usage chatUsage
	for _, text := range req.Texts {
		if err := r.Context().Err(); err != nil {
			writeError(w, http.StatusRequestTimeout, codeRequestCanceled, err.Error())
			return
		}
		out, spent, err := h.translateOne(r, req.From, req.To, text)
		if err != nil {
			writeTranslateErr(w, err)
			return
		}
		usage = usage.add(spent)
		results = append(results, out)
	}
	declareUsage(w, usage)
	writeJSON(w, http.StatusOK, batchResponse{Results: results})
}

// translateOne returns the translation and what the engine says it cost. A
// rejection costs nothing: none of the gates below reach a model.
func (h *Handler) translateOne(r *http.Request, fromRaw, toRaw, text string) (string, chatUsage, error) {
	reject := func(msg string) (string, chatUsage, error) {
		return "", chatUsage{}, &clientError{code: codeInvalidRequest, msg: msg}
	}
	// Preserve caller whitespace (MTran does not trim); only reject empty.
	if text == "" {
		return reject("text is required")
	}
	toRaw = strings.TrimSpace(toRaw)
	if toRaw == "" {
		return reject(msgToRequired)
	}
	to, ok := h.catalog.Resolve(toRaw)
	if !ok {
		return reject("unsupported target language for this model: " + toRaw)
	}

	fromRaw = strings.TrimSpace(fromRaw)
	autoFrom := fromRaw == "" || strings.EqualFold(fromRaw, "auto")
	var from Language
	if !autoFrom {
		from, ok = h.catalog.Resolve(fromRaw)
		if !ok {
			return reject("unsupported source language for this model: " + fromRaw)
		}
	}

	return h.Completer.Complete(r.Context(), h.prompt(from, to, text, autoFrom))
}

type clientError struct {
	code string
	msg  string
}

func (e *clientError) Error() string { return e.msg }

func writeRequestErr(w http.ResponseWriter, err error) {
	var ce *clientError
	if errors.As(err, &ce) {
		status := http.StatusBadRequest
		if ce.code == codePayloadTooLarge {
			status = http.StatusRequestEntityTooLarge
		}
		writeError(w, status, ce.code, ce.msg)
		return
	}
	writeError(w, http.StatusBadRequest, codeInvalidRequest, err.Error())
}

func writeTranslateErr(w http.ResponseWriter, err error) {
	var ce *clientError
	if errors.As(err, &ce) {
		writeError(w, http.StatusBadRequest, ce.code, ce.msg)
		return
	}
	writeError(w, http.StatusInternalServerError, codeTranslationErr, err.Error())
}

func decodeJSONBody(r *http.Request, dst any) error {
	defer r.Body.Close()
	limited := io.LimitReader(r.Body, maxBodyBytes+1)
	data, err := io.ReadAll(limited)
	if err != nil {
		return err
	}
	if len(data) > maxBodyBytes {
		return errBodyTooLarge
	}
	if len(strings.TrimSpace(string(data))) == 0 {
		return &clientError{code: codeInvalidRequest, msg: "request body is required"}
	}
	if err := json.Unmarshal(data, dst); err != nil {
		return err
	}
	return nil
}

var errBodyTooLarge = &clientError{code: codePayloadTooLarge, msg: "request body exceeds 1 MiB"}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	enc := json.NewEncoder(w)
	// Do not HTML-escape translation output (<, >, &).
	enc.SetEscapeHTML(false)
	_ = enc.Encode(v)
}

func writeError(w http.ResponseWriter, status int, code, msg string) {
	writeJSON(w, status, map[string]any{
		jsonKeyError: map[string]any{
			jsonKeyCode:    code,
			jsonKeyMessage: msg,
		},
	})
}
