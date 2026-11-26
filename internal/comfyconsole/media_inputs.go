package comfyconsole

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"strings"
)

const (
	assetMask             = "mask"
	assetImageKind        = "image"
	assetVideoKind        = "video"
	assetReferenceImage   = "ref_image"
	assetFirstFrame       = "first_frame"
	studioMediaImageLimit = 10 << 20
	studioMediaInputLimit = 20 << 20
	studioMediaBodyLimit  = 28 << 20
)

type generationInputs struct {
	Images []string `json:"images,omitempty"`
	Mask   string   `json:"mask,omitempty"`
	Video  string   `json:"video,omitempty"`
	Audio  string   `json:"audio,omitempty"`
}

type generationAsset struct {
	role      string
	kind      string
	extension string
	data      []byte
}

type studioAsset struct {
	ID   string `json:"id"`
	Role string `json:"role"`
}

func (flow APIWorkflow) imageLimit() int {
	if flow.MaxImages > 0 {
		return flow.MaxImages
	}
	return 1
}

func prepareGenerationInputs(inputs *generationInputs, flow APIWorkflow) ([]generationAsset, error) {
	if flow.Operation == operationGenerate {
		if inputs != nil {
			return nil, errors.New("text generation does not accept media inputs")
		}
		return nil, nil
	}
	if inputs == nil {
		inputs = &generationInputs{}
	}
	reference := flow.Operation == operationReference
	frames := flow.Operation == operationFirstFrame
	switch {
	case len(inputs.Images) > flow.imageLimit():
		return nil, fmt.Errorf("at most %d input images are allowed", flow.imageLimit())
	case reference && len(inputs.Images) == 0 && inputs.Video == "":
		return nil, errors.New("reference generation requires a reference image or video")
	case !reference && len(inputs.Images) == 0:
		return nil, errors.New("a source image or first frame is required")
	case !reference && (inputs.Video != "" || inputs.Audio != ""):
		return nil, errors.New("reference video and audio require the reference operation")
	case flow.Operation != operationEdit && inputs.Mask != "":
		return nil, errors.New("only image edits accept a mask")
	}

	remaining := studioMediaInputLimit
	var assets []generationAsset
	add := func(role, kind, value string, bare bool) error {
		if value == "" {
			return fmt.Errorf("%s is empty", role)
		}
		asset, err := decodeGenerationAsset(role, kind, value, bare, remaining)
		if err != nil {
			return err
		}
		assets = append(assets, asset)
		remaining -= len(asset.data)
		return nil
	}
	for index, image := range inputs.Images {
		role := assetReferenceImage
		if frames {
			role = assetFirstFrame
			if index == 1 {
				role = "last_frame"
			}
		}
		if err := add(role, assetImageKind, image, frames); err != nil {
			return nil, err
		}
	}
	for _, extra := range []struct{ role, kind, value string }{
		{assetMask, assetImageKind, inputs.Mask},
		{"ref_video", assetVideoKind, inputs.Video},
		{"ref_audio", "audio", inputs.Audio},
	} {
		if extra.value != "" {
			if err := add(extra.role, extra.kind, extra.value, false); err != nil {
				return nil, err
			}
		}
	}
	return assets, nil
}

var generationExtensions = map[string]string{
	"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp",
	"video/mp4": ".mp4", "video/quicktime": ".mov", "video/webm": ".webm",
	"audio/wav": ".wav", "audio/mpeg": ".mp3", "audio/flac": ".flac",
	"audio/ogg": ".ogg", "audio/aac": ".aac", "audio/mp4": ".m4a",
}

func decodeGenerationAsset(role, kind, value string, allowBare bool, remaining int) (generationAsset, error) {
	header, encoded, hasHeader := strings.Cut(value, ",")
	var declared string
	switch {
	case hasHeader && strings.HasPrefix(header, "data:"+kind+"/") && strings.HasSuffix(header, ";base64"):
		declared = strings.TrimSuffix(strings.TrimPrefix(header, "data:"), ";base64")
		if generationExtensions[declared] == "" {
			return generationAsset{}, fmt.Errorf("%s has an unsupported media type", role)
		}
	case allowBare && !hasHeader:
		encoded = value
	default:
		return generationAsset{}, fmt.Errorf("%s must be a Base64 %s data URL", role, kind)
	}
	limit := remaining
	if kind == assetImageKind {
		limit = min(limit, studioMediaImageLimit)
	}
	if len(encoded) > base64.StdEncoding.EncodedLen(limit) {
		return generationAsset{}, fmt.Errorf("%s exceeds the media size limit (10 MiB per image, 20 MiB combined)", role)
	}
	data, err := base64.StdEncoding.DecodeString(encoded)
	if err != nil {
		data, err = base64.RawStdEncoding.DecodeString(encoded)
	}
	if err != nil || len(data) == 0 || len(data) > limit {
		return generationAsset{}, fmt.Errorf("%s contains invalid or oversized media", role)
	}
	if kind == assetImageKind {
		actual := http.DetectContentType(data)
		if !strings.HasPrefix(actual, "image/") || generationExtensions[actual] == "" ||
			(declared != "" && declared != actual) {
			return generationAsset{}, fmt.Errorf("%s must contain a PNG, JPEG, or WebP image matching its declared type", role)
		}
		declared = actual
	}
	return generationAsset{role: role, kind: kind, extension: generationExtensions[declared], data: data}, nil
}

func (c *Console) uploadGenerationInputs(w http.ResponseWriter, r *http.Request, user string, inputs []generationAsset) ([]studioAsset, bool) {
	assets := make([]studioAsset, 0, len(inputs))
	for _, input := range inputs {
		var upload bytes.Buffer
		writer := multipart.NewWriter(&upload)
		part, err := writer.CreateFormFile("file", input.role+input.extension)
		if err == nil {
			_, err = part.Write(input.data)
		}
		if err == nil {
			err = writer.Close()
		}
		if err != nil {
			mediaError(w, http.StatusInternalServerError, "upload_failed", "Could not encode Studio asset")
			return nil, false
		}
		response := c.studioRequest(w, r, user, http.MethodPost, "/uploads", writer.FormDataContentType(), &upload)
		if response == nil {
			return nil, false
		}
		var asset struct {
			ID   string `json:"id"`
			Kind string `json:"kind"`
		}
		err = json.NewDecoder(io.LimitReader(response.Body, 1<<20)).Decode(&asset)
		_ = response.Body.Close()
		if err != nil || !studioIDPattern.MatchString(asset.ID) {
			mediaError(w, http.StatusBadGateway, "invalid_studio_response", "Studio did not return a valid asset")
			return nil, false
		}
		if asset.Kind != input.kind {
			mediaError(w, http.StatusBadRequest, "invalid_asset_type", fmt.Sprintf("%s must be a %s", input.role, input.kind))
			return nil, false
		}
		assets = append(assets, studioAsset{ID: asset.ID, Role: input.role})
	}
	return assets, true
}
