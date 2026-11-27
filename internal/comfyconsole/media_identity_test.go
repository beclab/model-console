package comfyconsole

import (
	"context"
	"crypto"
	"crypto/rand"
	"crypto/rsa"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"errors"
	"math/big"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"
)

func TestMediaIdentity(t *testing.T) {
	key, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		t.Fatal(err)
	}
	const issuer = "https://kubernetes.default.svc.cluster.local"
	const mediaAudience = mediaAudiencePrefix + "imageqwenimage21studio"
	var kid atomic.Value
	kid.Store("first")
	var requests atomic.Int32
	contentTypes := map[string]string{
		issuerDiscoveryPath: "application/json",
		issuerKeysPath:      "application/jwk-set+json",
	}
	api := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer model-service-account" {
			http.Error(w, "missing discovery credential", http.StatusUnauthorized)
			return
		}
		// Kubernetes negotiates the two endpoint media types separately.
		if contentType := contentTypes[r.URL.Path]; contentType != "" {
			if r.Header.Get("Accept") != contentType {
				http.Error(w, "unsupported Accept media type", http.StatusNotAcceptable)
				return
			}
			w.Header().Set("Content-Type", contentType)
		}
		requests.Add(1)
		switch r.URL.Path {
		case issuerDiscoveryPath:
			// Trust stays pinned to the API Service regardless of the advertised address.
			_ = json.NewEncoder(w).Encode(map[string]string{"issuer": issuer, "jwks_uri": "https://other.invalid/jwks"})
		case issuerKeysPath:
			_ = json.NewEncoder(w).Encode(map[string]any{"keys": []map[string]string{{
				"kid": kid.Load().(string), "kty": "RSA", "use": "sig", "alg": mediaJWTAlgorithm,
				"n": base64.RawURLEncoding.EncodeToString(key.N.Bytes()),
				"e": base64.RawURLEncoding.EncodeToString(big.NewInt(int64(key.E)).Bytes()),
			}}})
		default:
			http.NotFound(w, r)
		}
	}))
	defer api.Close()

	identity := &mediaIdentity{
		audience:  mediaAudience,
		origin:    api.URL,
		getClient: func() (*http.Client, error) { return api.Client(), nil },
		readToken: func() ([]byte, error) { return []byte("model-service-account"), nil },
	}
	now := time.Now().Unix()
	claims := map[string]any{
		"iss": issuer, "sub": routerMediaSubject, "aud": []string{mediaAudience},
		"iat": now - 30, "nbf": now - 30, "exp": now + 570,
	}
	jwt := func(kid, alg string, changes map[string]any) string {
		t.Helper()
		encode := func(value any) string {
			payload, err := json.Marshal(value)
			if err != nil {
				t.Fatal(err)
			}
			return base64.RawURLEncoding.EncodeToString(payload)
		}
		payload := make(map[string]any, len(claims))
		for k, v := range claims {
			payload[k] = v
		}
		for k, v := range changes {
			payload[k] = v
		}
		message := encode(map[string]string{"kid": kid, "alg": alg}) + "." + encode(payload)
		sum := sha256.Sum256([]byte(message))
		sig, err := rsa.SignPKCS1v15(rand.Reader, key, crypto.SHA256, sum[:])
		if err != nil {
			t.Fatal(err)
		}
		return message + "." + base64.RawURLEncoding.EncodeToString(sig)
	}
	check := func(t *testing.T, token string, want bool) {
		t.Helper()
		ok, err := identity.verify(context.Background(), token)
		if ok != want || err != nil {
			t.Fatalf("authorized = %t, err = %v, want %t", ok, err, want)
		}
	}
	check(t, jwt("first", mediaJWTAlgorithm, nil), true)
	for _, tc := range []struct {
		name, kid, alg string
		changes        map[string]any
	}{
		{"wrong issuer", "first", mediaJWTAlgorithm, map[string]any{"iss": "https://untrusted.example"}},
		{"wrong service account", "first", mediaJWTAlgorithm, map[string]any{"sub": "system:serviceaccount:router-shared:default"}},
		{"wrong namespace", "first", mediaJWTAlgorithm, map[string]any{"sub": "system:serviceaccount:other-shared:router-media"}},
		{"wrong audience", "first", mediaJWTAlgorithm, map[string]any{"aud": []string{issuer}}},
		{"another model", "first", mediaJWTAlgorithm, map[string]any{"aud": []string{mediaAudiencePrefix + "videominimaxh3studio"}}},
		{"multi-audience", "first", mediaJWTAlgorithm, map[string]any{"aud": []string{mediaAudience, issuer}}},
		{"expired", "first", mediaJWTAlgorithm, map[string]any{"exp": now - 1}},
		{"not yet valid", "first", mediaJWTAlgorithm, map[string]any{"nbf": now + 120}},
		{"long lifetime", "first", mediaJWTAlgorithm, map[string]any{"exp": now + 3600}},
		{"other algorithm", "first", "HS256", nil},
		{"unknown key", "wrong", mediaJWTAlgorithm, nil},
	} {
		t.Run(tc.name, func(t *testing.T) { check(t, jwt(tc.kid, tc.alg, tc.changes), false) })
	}
	check(t, jwt("first", mediaJWTAlgorithm, nil)+"tampered", false)
	if count := requests.Load(); count != 2 {
		t.Fatalf("Kubernetes API calls after repeated identities = %d, want 2", count)
	}

	// Keep accepting the current key during refresh throttling. Then fetch a
	// newly published key ID without waiting for the existing cache to expire.
	kid.Store("rotated")
	identity.nextRefresh = time.Time{}
	check(t, jwt("rotated", mediaJWTAlgorithm, nil), true)
	check(t, jwt("first", mediaJWTAlgorithm, nil), false)
	if count := requests.Load(); count != 3 {
		t.Fatalf("Kubernetes API calls after rotation = %d, want 3", count)
	}
	identity.expires = time.Time{}
	if ok, err := identity.verify(context.Background(), jwt("rotated", mediaJWTAlgorithm, nil)); ok || !errors.Is(err, errMediaTrustUnavailable) {
		t.Fatalf("expired key cache was usable: authorized = %t, err = %v", ok, err)
	}
}
