package hfwrap

import "testing"

func TestClassifyExit_Network(t *testing.T) {
	for _, snippet := range []string{
		"ConnectionError: 503 upstream busy",
		"requests.exceptions.HTTPError: 502 Bad Gateway",
		"http 504 gateway timeout",
		"Read timed out",
	} {
		t.Run(snippet, func(t *testing.T) {
			c, retry := classifyExit(1, snippet)
			if c != CodeNetwork {
				t.Errorf("got %d want CodeNetwork (%d)", c, CodeNetwork)
			}
			if !retry {
				t.Errorf("network errors must be retriable")
			}
		})
	}
}

func TestClassifyExit_RepoNotFound(t *testing.T) {
	c, retry := classifyExit(1, "huggingface_hub.errors.RepositoryNotFoundError: 404 not found for repo")
	if c != CodeRepoNotFound {
		t.Errorf("got %d want CodeRepoNotFound", c)
	}
	if retry {
		t.Errorf("repo_not_found must be permanent")
	}
}

func TestClassifyExit_RevisionNotFound(t *testing.T) {
	c, retry := classifyExit(1, "RevisionNotFoundError: revision 'abcdef' not found")
	if c != CodeRevision {
		t.Errorf("got %d want CodeRevision", c)
	}
	if retry {
		t.Errorf("revision-not-found must be permanent")
	}
}

func TestClassifyExit_Gated(t *testing.T) {
	c, retry := classifyExit(1, "huggingface_hub.errors.GatedRepoError: gated")
	if c != CodeGated {
		t.Errorf("got %d want CodeGated", c)
	}
	if retry {
		t.Errorf("gated must be permanent")
	}
}

func TestClassifyExit_TokenMissing(t *testing.T) {
	for _, snippet := range []string{
		"LocalTokenNotFoundError: not logged in",
		"401 Client Error: Unauthorized",
		"Invalid credentials in stored token",
	} {
		t.Run(snippet, func(t *testing.T) {
			c, retry := classifyExit(1, snippet)
			if c != CodeTokenMissing {
				t.Errorf("got %d want CodeTokenMissing", c)
			}
			if retry {
				t.Errorf("token_missing must be permanent")
			}
		})
	}
}

func TestClassifyExit_Permission(t *testing.T) {
	for _, snippet := range []string{
		"OSError: [Errno 13] Permission denied: /cache/hf/hub",
		"EACCES: cannot mkdir cache",
		"Read-only file system",
	} {
		t.Run(snippet, func(t *testing.T) {
			c, retry := classifyExit(1, snippet)
			if c != CodePermission {
				t.Errorf("got %d want CodePermission", c)
			}
			if retry {
				t.Errorf("permission errors must be permanent")
			}
		})
	}
}

func TestClassifyExit_DiskFull(t *testing.T) {
	c, retry := classifyExit(1, "OSError: [Errno 28] No space left on device")
	if c != CodeDiskFull {
		t.Errorf("got %d want CodeDiskFull", c)
	}
	if retry {
		t.Errorf("disk_full must be permanent")
	}
}

func TestClassifyExit_4xxFallsToInternal(t *testing.T) {
	c, retry := classifyExit(1, "huggingface_hub.errors.HfHubHTTPError: 422 Unprocessable Entity")
	if c != CodeInternal {
		t.Errorf("got %d want CodeInternal", c)
	}
	if retry {
		t.Errorf("4xx must be permanent")
	}
}

func TestClassifyExit_UnknownFallsToNetworkRetriable(t *testing.T) {
	c, retry := classifyExit(2, "completely unrecognized output")
	if c != CodeNetwork {
		t.Errorf("got %d want CodeNetwork (fallback)", c)
	}
	if !retry {
		t.Errorf("unknown fallback must be retriable")
	}
}

func TestClassifyExit_ZeroExitCode(t *testing.T) {
	c, retry := classifyExit(0, "")
	if c != CodeOK {
		t.Errorf("exit 0 must be CodeOK; got %d", c)
	}
	if retry {
		t.Errorf("CodeOK must not be retriable")
	}
}

func TestClassifyExit_NegativeExitCodeIsKilled(t *testing.T) {
	c, retry := classifyExit(-1, "")
	if c != CodeProcessKilled {
		t.Errorf("got %d want CodeProcessKilled", c)
	}
	if !retry {
		t.Errorf("CodeProcessKilled must be retriable")
	}
}
