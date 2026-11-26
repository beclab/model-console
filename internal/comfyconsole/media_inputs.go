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

	"github.com/llm-init/llm-init/internal/config"
)

const (
	assetReferenceImage   = "ref_image"
	assetFirstFrame       = "first_frame"
	studioMediaImageLimit = 10 << 20
	studioMediaBodyLimit  = 28 << 20
)

type generationInputs struct {
	Images []string `json:"images,omitempty"`
	Mask   string   `json:"mask,omitempty"`
}

type generationImage struct {
	role      string
	extension string
	data      []byte
}

type studioAsset struct {
	ID   string `json:"id"`
	Role string `json:"role"`
}

func prepareGenerationInputs(inputs *generationInputs, operation string) ([]generationImage, error) {
	if operation == operationGenerate {
		if inputs != nil {
			return nil, errors.New("text generation does not accept image inputs")
		}
		return nil, nil
	}
	if inputs == nil || len(inputs.Images) != 1 {
		return nil, errors.New("exactly one input image is required")
	}
	role := assetReferenceImage
	allowBare := operation == operationFirstFrame
	if allowBare {
		role = assetFirstFrame
		if inputs.Mask != "" {
			return nil, errors.New("frame generation does not accept a mask")
		}
	}
	image, err := decodeGenerationImage(role, inputs.Images[0], allowBare)
	if err != nil {
		return nil, err
	}
	images := []generationImage{image}
	if inputs.Mask != "" {
		mask, err := decodeGenerationImage("mask", inputs.Mask, false)
		if err != nil {
			return nil, err
		}
		images = append(images, mask)
	}
	return images, nil
}

func decodeGenerationImage(role, value string, allowBare bool) (generationImage, error) {
	header, encoded, hasHeader := strings.Cut(value, ",")
	var declared string
	switch {
	case hasHeader && strings.HasPrefix(header, "data:image/") && strings.HasSuffix(header, ";base64"):
		declared = strings.TrimSuffix(strings.TrimPrefix(header, "data:"), ";base64")
	case allowBare && !hasHeader:
		encoded = value
	default:
		return generationImage{}, fmt.Errorf("%s must be a Base64 image", role)
	}

	if len(encoded) > base64.StdEncoding.EncodedLen(studioMediaImageLimit) {
		return generationImage{}, fmt.Errorf("%s exceeds the 10 MiB image limit", role)
	}
	data, err := base64.StdEncoding.DecodeString(encoded)
	if err != nil {
		data, err = base64.RawStdEncoding.DecodeString(encoded)
	}
	if err != nil || len(data) == 0 || len(data) > studioMediaImageLimit {
		return generationImage{}, fmt.Errorf("%s contains an invalid or oversized image", role)
	}
	actual := http.DetectContentType(data)
	extension, ok := map[string]string{"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp"}[actual]
	if !ok || (declared != "" && declared != actual) {
		return generationImage{}, fmt.Errorf("%s must contain a PNG, JPEG, or WebP image matching its declared type", role)
	}
	return generationImage{role: role, extension: extension, data: data}, nil
}

func (c *Console) uploadGenerationInputs(w http.ResponseWriter, r *http.Request, user string, images []generationImage) ([]studioAsset, bool) {
	assets := make([]studioAsset, 0, len(images))
	for _, image := range images {
		var upload bytes.Buffer
		writer := multipart.NewWriter(&upload)
		part, err := writer.CreateFormFile("file", image.role+image.extension)
		if err == nil {
			_, err = part.Write(image.data)
		}
		if err == nil {
			err = writer.Close()
		}
		if err != nil {
			mediaError(w, http.StatusInternalServerError, "upload_failed", "Could not encode Studio image")
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
		if err != nil || asset.Kind != string(config.ModelImage) || !studioIDPattern.MatchString(asset.ID) {
			mediaError(w, http.StatusBadGateway, "invalid_studio_response", "Studio did not return a valid image asset")
			return nil, false
		}
		assets = append(assets, studioAsset{ID: asset.ID, Role: image.role})
	}
	return assets, true
}
