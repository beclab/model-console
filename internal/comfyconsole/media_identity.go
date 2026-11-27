package comfyconsole

import (
	"context"
	"crypto"
	"crypto/rsa"
	"crypto/sha256"
	"crypto/tls"
	"crypto/x509"
	"encoding/base64"
	"encoding/json"
	"errors"
	"io"
	"math/big"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"
)

const (
	mediaAuthSharedToken = "shared-token"
	mediaAuthKubernetes  = "kubernetes-jwks"
	mediaJWTAlgorithm    = "RS256"
	mediaAudiencePrefix  = "olares:model-console:media:"
	routerMediaSubject   = "system:serviceaccount:router-shared:router-media"
	kubeAPIOrigin        = "https://kubernetes.default.svc"
	kubeServiceAccount   = "/var/run/secrets/kubernetes.io/serviceaccount/"
	issuerDiscoveryPath  = "/.well-known/openid-configuration"
	issuerKeysPath       = "/openid/v1/jwks"
	keyCacheTTL          = 5 * time.Minute
	keyRefreshInterval   = 30 * time.Second
	mediaTokenLifetime   = 600
	mediaTokenClockSkew  = 30
)

var errMediaTrustUnavailable = errors.New("service account verification is unavailable")

type mediaIdentity struct {
	audience  string
	origin    string
	getClient func() (*http.Client, error)
	readToken func() ([]byte, error)

	mu          sync.Mutex
	client      *http.Client
	issuer      string
	keys        map[string]*rsa.PublicKey
	expires     time.Time
	nextRefresh time.Time
}

func newMediaIdentity(appName string) *mediaIdentity {
	return &mediaIdentity{
		audience: mediaAudiencePrefix + appName,
		origin:   kubeAPIOrigin,
		getClient: func() (*http.Client, error) {
			ca, err := os.ReadFile(kubeServiceAccount + "ca.crt")
			if err != nil {
				return nil, err
			}
			roots := x509.NewCertPool()
			if !roots.AppendCertsFromPEM(ca) {
				return nil, errors.New("invalid Kubernetes API CA")
			}
			// Connect directly to the in-cluster API, using only its mounted CA.
			return &http.Client{
				Transport: &http.Transport{TLSClientConfig: &tls.Config{
					RootCAs: roots, MinVersion: tls.VersionTLS12,
				}},
				Timeout: 3 * time.Second,
				CheckRedirect: func(*http.Request, []*http.Request) error {
					return http.ErrUseLastResponse
				},
			}, nil
		},
		readToken: func() ([]byte, error) { return os.ReadFile(kubeServiceAccount + "token") },
	}
}

func (v *mediaIdentity) verify(ctx context.Context, token string) (bool, error) {
	const maxJWTSize = 8192
	if len(token) == 0 || len(token) > maxJWTSize {
		return false, nil
	}
	parts := strings.Split(token, ".")
	if len(parts) != 3 {
		return false, nil
	}
	var header struct {
		Alg string `json:"alg"`
		Kid string `json:"kid"`
	}
	var claims struct {
		Iss string   `json:"iss"`
		Sub string   `json:"sub"`
		Aud []string `json:"aud"`
		Iat int64    `json:"iat"`
		Nbf int64    `json:"nbf"`
		Exp int64    `json:"exp"`
	}
	if !decodeJWTPart(parts[0], &header) || header.Alg != mediaJWTAlgorithm || header.Kid == "" ||
		!decodeJWTPart(parts[1], &claims) {
		return false, nil
	}
	now := time.Now().Unix()
	if claims.Sub != routerMediaSubject || len(claims.Aud) != 1 || claims.Aud[0] != v.audience ||
		claims.Iat <= 0 || claims.Iat > now+mediaTokenClockSkew ||
		claims.Nbf > now+mediaTokenClockSkew || claims.Exp <= now ||
		claims.Exp <= claims.Iat || claims.Exp-claims.Iat > mediaTokenLifetime {
		return false, nil
	}
	signature, err := base64.RawURLEncoding.DecodeString(parts[2])
	if err != nil {
		return false, nil
	}
	issuer, key, err := v.lookupKey(ctx, header.Kid)
	if err != nil {
		return false, err
	}
	if key == nil || claims.Iss != issuer {
		return false, nil
	}
	sum := sha256.Sum256([]byte(parts[0] + "." + parts[1]))
	return rsa.VerifyPKCS1v15(key, crypto.SHA256, sum[:], signature) == nil, nil
}

func decodeJWTPart(part string, value any) bool {
	data, err := base64.RawURLEncoding.DecodeString(part)
	return err == nil && json.Unmarshal(data, value) == nil
}

func (v *mediaIdentity) lookupKey(ctx context.Context, kid string) (string, *rsa.PublicKey, error) {
	v.mu.Lock()
	defer v.mu.Unlock()
	now := time.Now()
	current := now.Before(v.expires)
	if current && v.keys[kid] != nil {
		return v.issuer, v.keys[kid], nil
	}
	// Rate-limit unknown key IDs and temporary API outages.
	if !now.Before(v.nextRefresh) {
		v.nextRefresh = now.Add(keyRefreshInterval)
		if v.refresh(ctx) == nil {
			v.expires = time.Now().Add(keyCacheTTL)
			return v.issuer, v.keys[kid], nil
		}
	}
	if current {
		return v.issuer, nil, nil
	}
	return "", nil, errMediaTrustUnavailable
}

func (v *mediaIdentity) refresh(ctx context.Context) error {
	if v.client == nil {
		client, err := v.getClient()
		if err != nil {
			return err
		}
		v.client = client
	}
	token, err := v.readToken()
	if err != nil {
		return err
	}
	bearer := strings.TrimSpace(string(token))
	var metadata struct {
		Issuer string `json:"issuer"`
	}
	if v.issuer == "" {
		if err := v.getJSON(ctx, issuerDiscoveryPath, "application/json", bearer, &metadata); err != nil {
			return err
		}
		if !strings.HasPrefix(metadata.Issuer, "https://") {
			return errMediaTrustUnavailable
		}
	}

	var jwks struct {
		Keys []mediaJWK `json:"keys"`
	}
	// The signer is configured by the cluster. Keep both reads on the trusted
	// API Service and never follow a token-supplied URL or metadata redirect.
	if err := v.getJSON(ctx, issuerKeysPath, "application/jwk-set+json", bearer, &jwks); err != nil {
		return err
	}
	keys := make(map[string]*rsa.PublicKey, len(jwks.Keys))
	for _, jwk := range jwks.Keys {
		key := jwk.publicKey()
		if key == nil {
			continue
		}
		if keys[jwk.Kid] != nil {
			return errMediaTrustUnavailable
		}
		keys[jwk.Kid] = key
	}
	if len(keys) == 0 {
		return errMediaTrustUnavailable
	}
	if metadata.Issuer != "" {
		v.issuer = metadata.Issuer
	}
	v.keys = keys
	return nil
}

type mediaJWK struct {
	Kid string `json:"kid"`
	Kty string `json:"kty"`
	Use string `json:"use"`
	Alg string `json:"alg"`
	N   string `json:"n"`
	E   string `json:"e"`
}

func (jwk mediaJWK) publicKey() *rsa.PublicKey {
	if jwk.Kid == "" || jwk.Kty != "RSA" || (jwk.Alg != "" && jwk.Alg != mediaJWTAlgorithm) ||
		(jwk.Use != "" && jwk.Use != "sig") {
		return nil
	}
	n, nErr := base64.RawURLEncoding.DecodeString(jwk.N)
	e, eErr := base64.RawURLEncoding.DecodeString(jwk.E)
	if nErr != nil || eErr != nil || len(n) < 256 || len(n) > 1024 || len(e) == 0 || len(e) > 4 {
		return nil
	}
	modulus := new(big.Int).SetBytes(n)
	exponent := new(big.Int).SetBytes(e).Int64()
	if modulus.BitLen() < 2048 || modulus.Bit(0) != 1 || exponent < 3 || exponent > 1<<31-1 || exponent%2 != 1 {
		return nil
	}
	return &rsa.PublicKey{N: modulus, E: int(exponent)}
}

func (v *mediaIdentity) getJSON(ctx context.Context, path, accept, token string, value any) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, v.origin+path, nil)
	if err != nil {
		return err
	}
	req.Header.Set("Authorization", "Bearer "+token)
	req.Header.Set("Accept", accept)
	res, err := v.client.Do(req)
	if err != nil {
		return err
	}
	defer res.Body.Close()
	if res.StatusCode != http.StatusOK {
		return errMediaTrustUnavailable
	}
	const limit = 64 << 10
	data, err := io.ReadAll(io.LimitReader(res.Body, limit+1))
	if err != nil || len(data) > limit {
		return errMediaTrustUnavailable
	}
	return json.Unmarshal(data, value)
}
