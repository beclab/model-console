package comfyconsole

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

var (
	ErrNoGroups     = errors.New("this app has no model groups")
	ErrUnknownGroup = errors.New("unknown model group")
	ErrLastGroup    = errors.New("at least one model group must stay installed")
)

type selectionFile struct {
	Groups []string `json:"groups"`
}

// LoadSelection returns the groups to keep installed. The selection file
// (on persistent storage) wins; without one, initial is used: a comma list
// of group ids, "all", or empty for the first group. The result is written
// back so later starts keep what the user changed.
func LoadSelection(cfg *AppConfig, file, initial string) ([]string, error) {
	if len(cfg.Groups) == 0 {
		return nil, nil
	}
	var ids []string
	if file != "" {
		if raw, err := os.ReadFile(file); err == nil {
			var sf selectionFile
			if err := json.Unmarshal(raw, &sf); err != nil {
				return nil, fmt.Errorf("selection file %s: %w", file, err)
			}
			for _, id := range sf.Groups {
				if cfg.group(id) != nil {
					ids = append(ids, id)
				}
			}
		} else if !errors.Is(err, os.ErrNotExist) {
			return nil, err
		}
	}
	if len(ids) == 0 {
		switch initial = strings.TrimSpace(initial); initial {
		case "":
			ids = []string{cfg.Groups[0].ID}
		case "all":
			for _, g := range cfg.Groups {
				ids = append(ids, g.ID)
			}
		default:
			for _, id := range strings.Split(initial, ",") {
				id = strings.TrimSpace(id)
				if cfg.group(id) == nil {
					return nil, fmt.Errorf("COMFY_GROUPS: %w %q", ErrUnknownGroup, id)
				}
				ids = append(ids, id)
			}
		}
	}
	return ids, saveSelection(file, ids)
}

func saveSelection(file string, ids []string) error {
	if file == "" {
		return nil
	}
	ids = append([]string(nil), ids...)
	sort.Strings(ids)
	raw, err := json.Marshal(selectionFile{Groups: ids})
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(file), 0o755); err != nil {
		return err
	}
	tmp := file + ".tmp"
	if err := os.WriteFile(tmp, raw, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, file)
}

// wanted reports whether an item is kept: shared, or in a selected group.
// Callers hold c.mu.
func (c *Console) wanted(it *itemState) bool {
	if len(it.groups) == 0 {
		return true
	}
	for _, g := range it.groups {
		if c.selected[g] {
			return true
		}
	}
	return false
}

func (c *Console) selectedIDs() []string {
	var ids []string
	for _, g := range c.opts.Config.Groups {
		if c.selected[g.ID] {
			ids = append(ids, g.ID)
		}
	}
	return ids
}

// AddGroup selects a group; its models download in the next pass.
func (c *Console) AddGroup(id string) error {
	if len(c.opts.Config.Groups) == 0 {
		return ErrNoGroups
	}
	if c.opts.Config.group(id) == nil {
		return ErrUnknownGroup
	}
	c.mu.Lock()
	if c.selected[id] {
		c.mu.Unlock()
		return nil
	}
	c.selected[id] = true
	err := saveSelection(c.opts.SelectionFile, c.selectedIDs())
	c.mu.Unlock()
	c.opts.Logger.Info("model group added", "group", id)
	c.wake()
	return err
}

// RemoveGroup unselects a group and deletes its files that no other
// selected group uses (shared models stay). A download in progress is
// stopped first.
func (c *Console) RemoveGroup(id string) error {
	if len(c.opts.Config.Groups) == 0 {
		return ErrNoGroups
	}
	if c.opts.Config.group(id) == nil {
		return ErrUnknownGroup
	}
	c.mu.Lock()
	if !c.selected[id] {
		c.mu.Unlock()
		return nil
	}
	if len(c.selectedIDs()) == 1 {
		c.mu.Unlock()
		return ErrLastGroup
	}
	delete(c.selected, id)
	err := saveSelection(c.opts.SelectionFile, c.selectedIDs())
	if c.passCancel != nil {
		c.passCancel()
	}
	c.mu.Unlock()

	c.passMu.Lock()
	for _, it := range c.items {
		c.mu.Lock()
		drop := !c.wanted(it) && contains(it.groups, id)
		c.mu.Unlock()
		if !drop {
			continue
		}
		dest := filepath.Join(c.opts.ModelsDir, filepath.FromSlash(it.model.RelPath()))
		for _, p := range []string{dest, dest + ".part", dest + ".part.json",
			filepath.Join(c.opts.ModelsDir, markerDir, filepath.FromSlash(it.model.RelPath())+".json")} {
			if rerr := os.Remove(p); rerr != nil && !errors.Is(rerr, os.ErrNotExist) {
				c.opts.Logger.Warn("remove model file", "path", p, "err", rerr)
			}
		}
		c.set(it, StatePending, 0, "")
	}
	c.passMu.Unlock()
	c.opts.Logger.Info("model group removed", "group", id)
	c.wake()
	return err
}

func (c *Console) wake() {
	select {
	case c.kick <- struct{}{}:
	default:
	}
}

func contains(xs []string, x string) bool {
	for _, y := range xs {
		if y == x {
			return true
		}
	}
	return false
}
