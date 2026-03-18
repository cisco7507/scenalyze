import { Fragment, useDeferredValue, useEffect, useMemo, useState } from 'react';
import type { ReactElement } from 'react';
import {
  ChevronDownIcon,
  ChevronRightIcon,
  Cross2Icon,
  InfoCircledIcon,
  LayersIcon,
  MagnifyingGlassIcon,
  UpdateIcon,
} from '@radix-ui/react-icons';
import {
  getTaxonomyExplorer,
} from '../lib/api';
import type {
  TaxonomyExplorerGroup,
  TaxonomyExplorerItem,
  TaxonomyExplorerResponse,
} from '../lib/api';

type ExplorerSelection =
  | { kind: 'group'; group: TaxonomyExplorerGroup }
  | { kind: 'item'; item: TaxonomyExplorerItem };

function normalizeSearch(value: string): string {
  return value.trim().toLowerCase();
}

function renderHighlightedText(value: string, query: string) {
  if (!query) return value;
  const lower = value.toLowerCase();
  const queryLower = query.toLowerCase();
  const segments: Array<string | { text: string; match: true }> = [];
  let cursor = 0;

  while (cursor < value.length) {
    const nextIndex = lower.indexOf(queryLower, cursor);
    if (nextIndex === -1) {
      segments.push(value.slice(cursor));
      break;
    }
    if (nextIndex > cursor) {
      segments.push(value.slice(cursor, nextIndex));
    }
    segments.push({ text: value.slice(nextIndex, nextIndex + queryLower.length), match: true });
    cursor = nextIndex + queryLower.length;
  }

  return segments.map((segment, index) =>
    typeof segment === 'string' ? (
      <Fragment key={`${value}-${index}`}>{segment}</Fragment>
    ) : (
      <mark
        key={`${value}-${index}`}
        className="rounded bg-[#fef08a] px-0.5 text-inherit"
      >
        {segment.text}
      </mark>
    ),
  );
}

function statValueClass(tone: 'slate' | 'blue' | 'green' | 'violet') {
  if (tone === 'blue') return 'text-primary-700';
  if (tone === 'green') return 'text-emerald-600';
  if (tone === 'violet') return 'text-violet-700';
  return 'text-slate-950';
}

function ExplorerStatCard({
  label,
  value,
  detail,
  tone,
}: {
  label: string;
  value: string | number;
  detail: string;
  tone: 'slate' | 'blue' | 'green' | 'violet';
}) {
  return (
    <div className="rounded-[1.7rem] border border-slate-200/90 bg-white px-5 py-4 shadow-[0_16px_30px_rgba(0,55,120,0.06)]">
      <div className="text-[10px] font-semibold uppercase tracking-[0.22em] text-slate-400">{label}</div>
      <div className={`mt-3 text-[2rem] font-bold ${statValueClass(tone)}`}>{value}</div>
      <div className="mt-2 text-xs text-slate-500">{detail}</div>
    </div>
  );
}

export function TaxonomyExplorer() {
  const [payload, setPayload] = useState<TaxonomyExplorerResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [search, setSearch] = useState('');
  const [expandedGroupIds, setExpandedGroupIds] = useState<Set<string>>(new Set());
  const [expandedItemIds, setExpandedItemIds] = useState<Set<string>>(new Set());
  const [selection, setSelection] = useState<ExplorerSelection | null>(null);

  const deferredSearch = useDeferredValue(search);
  const normalizedQuery = normalizeSearch(deferredSearch);
  const searchActive = normalizedQuery.length > 0;

  const fetchExplorer = async () => {
    setLoading(true);
    try {
      const nextPayload = await getTaxonomyExplorer();
      setPayload(nextPayload);
      setError('');
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to load taxonomy explorer.';
      setError(message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void fetchExplorer();
  }, []);

  const itemById = useMemo(() => {
    const map = new Map<string, TaxonomyExplorerItem>();
    for (const item of payload?.items || []) {
      map.set(String(item.id), item);
    }
    return map;
  }, [payload]);

  const childrenByParent = useMemo(() => {
    const map = new Map<string, TaxonomyExplorerItem[]>();
    for (const item of payload?.items || []) {
      const parentId = String(item.parent_id || '0');
      const bucket = map.get(parentId) || [];
      bucket.push(item);
      map.set(parentId, bucket);
    }
    for (const [parentId, items] of map.entries()) {
      map.set(
        parentId,
        [...items].sort((left, right) => {
          const byName = left.name.localeCompare(right.name, undefined, { sensitivity: 'base' });
          if (byName !== 0) return byName;
          return String(left.id).localeCompare(String(right.id));
        }),
      );
    }
    return map;
  }, [payload]);

  const childCountById = useMemo(() => {
    const counts = new Map<string, number>();
    for (const [parentId, items] of childrenByParent.entries()) {
      if (parentId === '0') continue;
      counts.set(parentId, items.length);
    }
    return counts;
  }, [childrenByParent]);

  const derivedSearch = useMemo(() => {
    const matchedItemIds = new Set<string>();
    const visibleItemIds = new Set<string>();
    const matchedGroupIds = new Set<string>();
    const matchedGroupChildIds = new Set<string>();

    if (!payload || !searchActive) {
      return { matchedItemIds, visibleItemIds, matchedGroupIds, matchedGroupChildIds };
    }

    for (const item of payload.items) {
      const itemId = String(item.id);
      if (
        itemId.toLowerCase().includes(normalizedQuery) ||
        item.name.toLowerCase().includes(normalizedQuery) ||
        item.path_text.toLowerCase().includes(normalizedQuery)
      ) {
        matchedItemIds.add(itemId);
        for (const pathId of item.path_ids) {
          visibleItemIds.add(String(pathId));
        }
      }
    }

    for (const group of payload.groups) {
      if (
        String(group.id).toLowerCase().includes(normalizedQuery) ||
        group.name.toLowerCase().includes(normalizedQuery)
      ) {
        matchedGroupIds.add(String(group.id));
      }
      for (const child of group.children) {
        const childId = String(child.id);
        if (
          childId.toLowerCase().includes(normalizedQuery) ||
          child.name.toLowerCase().includes(normalizedQuery)
        ) {
          matchedGroupChildIds.add(childId);
          let currentId = childId;
          while (currentId && currentId !== '0') {
            visibleItemIds.add(currentId);
            const nextItem = itemById.get(currentId);
            currentId = nextItem?.parent_id ? String(nextItem.parent_id) : '0';
          }
        }
      }
    }

    return { matchedItemIds, visibleItemIds, matchedGroupIds, matchedGroupChildIds };
  }, [itemById, normalizedQuery, payload, searchActive]);

  useEffect(() => {
    if (!payload) return;
    if (selection) {
      if (selection.kind === 'item' && itemById.has(String(selection.item.id))) return;
      if (
        selection.kind === 'group' &&
        payload.groups.some((group) => String(group.id) === String(selection.group.id))
      ) {
        return;
      }
    }

    const firstRootItem = childrenByParent.get('0')?.[0] || payload.items[0];
    if (firstRootItem) {
      setSelection({ kind: 'item', item: firstRootItem });
      return;
    }
    if (payload.groups[0]) {
      setSelection({ kind: 'group', group: payload.groups[0] });
    }
  }, [childrenByParent, itemById, payload, selection]);

  const visibleRootItems = useMemo(() => {
    const roots = childrenByParent.get('0') || [];
    if (!searchActive) return roots;
    return roots.filter((item) => derivedSearch.visibleItemIds.has(String(item.id)));
  }, [childrenByParent, derivedSearch.visibleItemIds, searchActive]);

  const matchingGroupCount = useMemo(() => {
    if (!payload) return 0;
    if (!searchActive) return payload.groups.length;
    return payload.groups.filter((group) => {
      const groupId = String(group.id);
      const groupMatches = derivedSearch.matchedGroupIds.has(groupId);
      if (groupMatches) return true;
      return group.children.some((child) => {
        const childId = String(child.id);
        return (
          derivedSearch.matchedGroupChildIds.has(childId) ||
          derivedSearch.visibleItemIds.has(childId)
        );
      });
    }).length;
  }, [
    derivedSearch.matchedGroupChildIds,
    derivedSearch.matchedGroupIds,
    derivedSearch.visibleItemIds,
    payload,
    searchActive,
  ]);

  const matchingItemCount = useMemo(() => {
    if (!payload) return 0;
    return searchActive ? derivedSearch.matchedItemIds.size : payload.items.length;
  }, [derivedSearch.matchedItemIds.size, payload, searchActive]);

  const selectedItem =
    selection?.kind === 'item' ? selection.item : null;
  const selectedItemParent =
    selectedItem && selectedItem.parent_id && selectedItem.parent_id !== '0'
      ? itemById.get(String(selectedItem.parent_id)) || null
      : null;

  const toggleGroup = (groupId: string) => {
    setExpandedGroupIds((current) => {
      const next = new Set(current);
      if (next.has(groupId)) next.delete(groupId);
      else next.add(groupId);
      return next;
    });
  };

  const toggleItem = (itemId: string) => {
    setExpandedItemIds((current) => {
      const next = new Set(current);
      if (next.has(itemId)) next.delete(itemId);
      else next.add(itemId);
      return next;
    });
  };

  const renderItemBranch = (
    parentId: string,
    depth = 0,
    forceShow = false,
  ): ReactElement | null => {
    const children = childrenByParent.get(parentId) || [];
    const visibleChildren = children.filter((item) => {
      if (!searchActive) return true;
      if (forceShow) return true;
      return derivedSearch.visibleItemIds.has(String(item.id));
    });

    if (visibleChildren.length === 0) return null;

    return (
      <div className={depth > 0 ? 'ml-4 border-l border-slate-200 pl-4' : ''}>
        {visibleChildren.map((item) => {
          const itemId = String(item.id);
          const directMatch = derivedSearch.matchedItemIds.has(itemId);
          const childTree = renderItemBranch(
            itemId,
            depth + 1,
            forceShow || directMatch,
          );
          const hasChildren = Boolean(childrenByParent.get(itemId)?.length);
          const autoExpanded =
            searchActive && (forceShow || directMatch || Boolean(childTree));
          const isExpanded = hasChildren && (autoExpanded || expandedItemIds.has(itemId));
          const isSelected = selection?.kind === 'item' && String(selection.item.id) === itemId;

          return (
            <div key={`item-${itemId}`} className="py-1">
              <div
                className={`flex items-start gap-2 rounded-xl px-2 py-1.5 transition-colors ${
                  isSelected ? 'bg-primary-50' : 'hover:bg-slate-50'
                }`}
              >
                {hasChildren ? (
                  <button
                    type="button"
                    onClick={() => toggleItem(itemId)}
                    className="mt-[2px] inline-flex h-5 w-5 shrink-0 items-center justify-center rounded-full border border-slate-200 bg-white text-slate-500 transition-colors hover:border-primary-200 hover:text-primary-700"
                    aria-label={isExpanded ? `Collapse ${item.name}` : `Expand ${item.name}`}
                  >
                    {isExpanded ? <ChevronDownIcon className="h-3.5 w-3.5" /> : <ChevronRightIcon className="h-3.5 w-3.5" />}
                  </button>
                ) : (
                  <span className="mt-2 inline-block h-1.5 w-1.5 shrink-0 rounded-full bg-slate-300" />
                )}

                <button
                  type="button"
                  onClick={() => setSelection({ kind: 'item', item })}
                  className="min-w-0 flex-1 text-left"
                >
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="text-sm font-medium text-slate-800">
                      {renderHighlightedText(item.name, normalizedQuery)}
                    </span>
                    <span className="inline-flex items-center rounded-full border border-primary-200 bg-primary-50 px-2 py-0.5 text-[10px] font-mono font-semibold text-primary-700">
                      ID {itemId}
                    </span>
                    {directMatch ? (
                      <span className="inline-flex items-center rounded-full border border-amber-200 bg-amber-50 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.14em] text-amber-700">
                        Match
                      </span>
                    ) : null}
                  </div>
                </button>
              </div>
              {isExpanded ? childTree : null}
            </div>
          );
        })}
      </div>
    );
  };

  const renderGroupTree = () => {
    if (!payload) return null;
    const groups = payload.groups.filter((group) => {
      const groupId = String(group.id);
      const groupMatches = derivedSearch.matchedGroupIds.has(groupId);
      if (!searchActive) return true;
      if (groupMatches) return true;
      return group.children.some((child) => {
        const childId = String(child.id);
        return (
          derivedSearch.matchedGroupChildIds.has(childId) ||
          derivedSearch.visibleItemIds.has(childId)
        );
      });
    });

    if (groups.length === 0) {
      return (
        <div className="rounded-[1.5rem] border border-dashed border-slate-200 bg-slate-50 px-4 py-10 text-sm text-slate-500">
          No industry groups match this search.
        </div>
      );
    }

    return groups.map((group) => {
      const groupId = String(group.id);
      const groupMatches = derivedSearch.matchedGroupIds.has(groupId);
      const visibleChildren = group.children.filter((child) => {
        if (!searchActive) return true;
        const childId = String(child.id);
        return (
          groupMatches ||
          derivedSearch.matchedGroupChildIds.has(childId) ||
          derivedSearch.visibleItemIds.has(childId)
        );
      });
      const autoExpanded = searchActive && (groupMatches || visibleChildren.length > 0);
      const isExpanded = autoExpanded || expandedGroupIds.has(groupId);
      const isSelected = selection?.kind === 'group' && String(selection.group.id) === groupId;

      return (
        <div key={`group-${groupId}`} className="py-2">
          <div
            className={`rounded-[1.2rem] border px-3 py-3 transition-colors ${
              isSelected
                ? 'border-emerald-200 bg-emerald-50/70'
                : 'border-slate-200 bg-white hover:border-emerald-200 hover:bg-emerald-50/40'
            }`}
          >
            <div className="flex items-start gap-3">
              <button
                type="button"
                onClick={() => toggleGroup(groupId)}
                className="mt-0.5 inline-flex h-6 w-6 shrink-0 items-center justify-center rounded-full border border-slate-200 bg-white text-slate-500 transition-colors hover:border-emerald-200 hover:text-emerald-700"
                aria-label={isExpanded ? `Collapse ${group.name}` : `Expand ${group.name}`}
              >
                {isExpanded ? <ChevronDownIcon className="h-4 w-4" /> : <ChevronRightIcon className="h-4 w-4" />}
              </button>
              <button
                type="button"
                onClick={() => setSelection({ kind: 'group', group })}
                className="min-w-0 flex-1 text-left"
              >
                <div className="flex flex-wrap items-center gap-2">
                  <span className="text-sm font-semibold text-slate-800">
                    {renderHighlightedText(group.name, normalizedQuery)}
                  </span>
                  {group.id ? (
                    <span className="inline-flex items-center rounded-full border border-emerald-200 bg-emerald-50 px-2 py-0.5 text-[10px] font-mono font-semibold text-emerald-700">
                      ID {group.id}
                    </span>
                  ) : null}
                  {groupMatches ? (
                    <span className="inline-flex items-center rounded-full border border-amber-200 bg-amber-50 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.14em] text-amber-700">
                      Match
                    </span>
                  ) : null}
                </div>
                <div className="mt-1 text-xs text-slate-500">
                  {visibleChildren.length} linked industr{visibleChildren.length === 1 ? 'y' : 'ies'}
                </div>
              </button>
            </div>

            {isExpanded ? (
              <div className="mt-3 space-y-2 border-t border-slate-200/80 pt-3">
                {visibleChildren.map((child) => {
                  const childId = String(child.id);
                  const childMatches = derivedSearch.matchedGroupChildIds.has(childId);
                  const linkedItem = itemById.get(childId);
                  const forceShow = groupMatches || childMatches;
                  const subtree = renderItemBranch(childId, 1, forceShow);
                  const isChildSelected = selection?.kind === 'item' && String(selection.item.id) === childId;

                  return (
                    <div key={`group-child-${groupId}-${childId}`} className="rounded-[1rem] border border-slate-200/80 bg-slate-50/75 px-3 py-2">
                      <button
                        type="button"
                        onClick={() =>
                          linkedItem
                            ? setSelection({ kind: 'item', item: linkedItem })
                            : setSelection({ kind: 'group', group })
                        }
                        className={`w-full text-left ${
                          isChildSelected ? 'text-primary-700' : 'text-slate-700'
                        }`}
                      >
                        <div className="flex flex-wrap items-center gap-2">
                          <span className="text-sm font-medium">
                            {renderHighlightedText(child.name, normalizedQuery)}
                          </span>
                          <span className="inline-flex items-center rounded-full border border-primary-200 bg-white px-2 py-0.5 text-[10px] font-mono font-semibold text-primary-700">
                            ID {childId}
                          </span>
                        </div>
                      </button>
                      {subtree ? <div className="mt-2">{subtree}</div> : null}
                    </div>
                  );
                })}
              </div>
            ) : null}
          </div>
        </div>
      );
    });
  };

  const selectedPanel = (
    <aside className="bell-panel-muted p-5">
      <div className="flex items-center justify-between gap-3">
        <div>
          <div className="text-[11px] font-semibold uppercase tracking-[0.22em] text-slate-400">
            Selection
          </div>
          <div className="mt-2 text-xl font-bold text-slate-900">
            {selection?.kind === 'group'
              ? selection.group.name
              : selection?.kind === 'item'
                ? selection.item.name
                : 'Select a category'}
          </div>
        </div>
        {selection ? (
          <button
            type="button"
            onClick={() => setSelection(null)}
            className="inline-flex h-9 w-9 items-center justify-center rounded-full border border-slate-200 bg-white text-slate-500 transition-colors hover:border-primary-200 hover:text-primary-700"
            aria-label="Clear selection"
          >
            <Cross2Icon className="h-4 w-4" />
          </button>
        ) : null}
      </div>

      {selection?.kind === 'group' ? (
        <div className="mt-5 space-y-3">
          <div className="rounded-[1.25rem] border border-slate-200 bg-white px-4 py-3">
            <div className="text-[10px] font-semibold uppercase tracking-[0.18em] text-slate-400">Group ID</div>
            <div className="mt-2 text-sm font-medium text-slate-800">{selection.group.id || '—'}</div>
          </div>
          <div className="rounded-[1.25rem] border border-slate-200 bg-white px-4 py-3">
            <div className="text-[10px] font-semibold uppercase tracking-[0.18em] text-slate-400">Linked industries</div>
            <div className="mt-2 text-sm font-medium text-slate-800">{selection.group.children.length}</div>
          </div>
          <div className="rounded-[1.25rem] border border-slate-200 bg-white px-4 py-3">
            <div className="text-[10px] font-semibold uppercase tracking-[0.18em] text-slate-400">Bridge children</div>
            <div className="mt-3 flex flex-wrap gap-2">
              {selection.group.children.length > 0 ? (
                selection.group.children.map((child) => (
                  <span
                    key={`selection-child-${child.id}`}
                    className="inline-flex items-center rounded-full border border-primary-200 bg-primary-50 px-2.5 py-1 text-[11px] font-medium text-primary-700"
                  >
                    {child.name}
                  </span>
                ))
              ) : (
                <span className="text-sm text-slate-500">No linked industries</span>
              )}
            </div>
          </div>
        </div>
      ) : selectedItem ? (
        <div className="mt-5 space-y-3">
          <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-1">
            {[
              { label: 'Category ID', value: selectedItem.id },
              { label: 'Level', value: String(selectedItem.level) },
              { label: 'Industry', value: `${selectedItem.industry_name} · ID ${selectedItem.industry_id}` },
              {
                label: 'Parent',
                value: selectedItemParent
                  ? `${selectedItemParent.name} · ID ${selectedItemParent.id}`
                  : 'Top-level category',
              },
              {
                label: 'Children',
                value: String(childCountById.get(String(selectedItem.id)) || 0),
              },
            ].map((field) => (
              <div key={field.label} className="rounded-[1.25rem] border border-slate-200 bg-white px-4 py-3">
                <div className="text-[10px] font-semibold uppercase tracking-[0.18em] text-slate-400">{field.label}</div>
                <div className="mt-2 text-sm font-medium text-slate-800">{field.value}</div>
              </div>
            ))}
          </div>
          <div className="rounded-[1.25rem] border border-slate-200 bg-white px-4 py-3">
            <div className="text-[10px] font-semibold uppercase tracking-[0.18em] text-slate-400">Full path</div>
            <div className="mt-2 text-sm leading-6 text-slate-700">{selectedItem.path_text}</div>
          </div>
          <div className="rounded-[1.25rem] border border-slate-200 bg-white px-4 py-3">
            <div className="text-[10px] font-semibold uppercase tracking-[0.18em] text-slate-400">Path IDs</div>
            <div className="mt-3 flex flex-wrap gap-2">
              {selectedItem.path_ids.map((pathId, index) => (
                <span
                  key={`${selectedItem.id}-${pathId}-${index}`}
                  className="inline-flex items-center rounded-full border border-slate-200 bg-slate-50 px-2.5 py-1 text-[11px] font-mono text-slate-700"
                >
                  {pathId}
                </span>
              ))}
            </div>
          </div>
        </div>
      ) : (
        <div className="mt-5 rounded-[1.5rem] border border-dashed border-slate-200 bg-white px-4 py-8 text-sm text-slate-500">
          Select a node from either tree to inspect its IDs, parent, industry, and full path.
        </div>
      )}

      <div className="mt-5 rounded-[1.25rem] border border-primary-100 bg-primary-50/75 px-4 py-3 text-sm text-slate-600">
        <div className="flex items-start gap-3">
          <InfoCircledIcon className="mt-0.5 h-4 w-4 shrink-0 text-primary-500" />
          <div>
            Search keeps matching categories visible together with their ancestor path, so you can see where a match lives in the taxonomy instead of only seeing isolated leaf nodes.
          </div>
        </div>
      </div>
    </aside>
  );

  return (
    <div className="space-y-6 animate-in fade-in duration-500">
      <section className="bell-hero">
        <div className="relative z-10 flex flex-col gap-6 xl:flex-row xl:items-end xl:justify-between">
          <div className="max-w-3xl">
            <div className="bell-badge">
              <LayersIcon className="h-3.5 w-3.5" />
              Loaded taxonomy
            </div>
            <h2 className="mt-4 max-w-3xl text-[3rem] font-bold text-primary-700">
              Explore the FreeWheel category map behind the classifier.
            </h2>
            <p className="mt-3 max-w-3xl text-sm leading-6 text-slate-600">
              Search industry groups, inspect canonical category paths, and follow how top-level industries bridge into the deeper taxonomy the mapper actually uses.
            </p>
            <div className="mt-6 flex flex-wrap items-center gap-3">
              <div className="bell-data-pill">Source {payload?.json_path_used.split('/').pop() || 'freewheel.json'}</div>
              <div className="bell-data-pill">Search preserves ancestor paths</div>
            </div>
          </div>

          <div className="grid gap-3 sm:grid-cols-2 xl:w-[26rem]">
            <ExplorerStatCard
              label="Industry groups"
              value={payload?.stats.group_count ?? '—'}
              detail="Top-level group bridges defined in the JSON."
              tone="slate"
            />
            <ExplorerStatCard
              label="Categories"
              value={payload?.stats.item_count ?? '—'}
              detail="Normalized items available to the taxonomy explorer."
              tone="blue"
            />
            <ExplorerStatCard
              label="Leaf categories"
              value={payload?.stats.leaf_count ?? '—'}
              detail="Terminal nodes with no children beneath them."
              tone="green"
            />
            <ExplorerStatCard
              label="Depth"
              value={payload ? `L${payload.stats.max_level}` : '—'}
              detail="Deepest level found in the loaded hierarchy."
              tone="violet"
            />
          </div>
        </div>
      </section>

      {loading ? (
        <div className="bell-panel px-6 py-10 text-sm text-slate-500">Loading taxonomy explorer…</div>
      ) : error ? (
        <div className="rounded-[24px] border border-rose-200 bg-rose-50/90 p-4 text-rose-700 shadow-sm">
          Failed to load taxonomy explorer: {error}
        </div>
      ) : payload && !payload.enabled ? (
        <div className="rounded-[24px] border border-rose-200 bg-rose-50/90 p-4 text-rose-700 shadow-sm">
          Taxonomy explorer unavailable. {payload.last_error || 'The taxonomy JSON could not be loaded.'}
        </div>
      ) : payload ? (
        <div className="grid gap-6 xl:grid-cols-[minmax(0,1.8fr)_22rem]">
          <section className="bell-panel overflow-hidden">
            <div className="border-b border-slate-200/80 bg-primary-50/75 px-6 py-4">
              <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
                <div>
                  <h3 className="text-lg font-bold text-slate-900">Taxonomy explorer</h3>
                  <p className="mt-1 text-sm text-slate-500">
                    Search once, then inspect both the group bridge and the deep item hierarchy side by side.
                  </p>
                </div>
                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={() => {
                      setSearch('');
                      setExpandedGroupIds(new Set());
                      setExpandedItemIds(new Set());
                    }}
                    className="bell-button-secondary h-10 gap-2 px-4 text-xs uppercase tracking-[0.18em]"
                  >
                    <Cross2Icon className="h-3.5 w-3.5" />
                    Clear
                  </button>
                  <button
                    type="button"
                    onClick={() => void fetchExplorer()}
                    className="inline-flex h-10 w-10 items-center justify-center rounded-full border border-slate-300 bg-white text-primary-700 shadow-sm transition-colors hover:border-primary-300 hover:bg-primary-50"
                    title="Refresh taxonomy data"
                  >
                    <UpdateIcon className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
                  </button>
                </div>
              </div>

              <div className="mt-4 flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
                <label className="relative block flex-1">
                  <MagnifyingGlassIcon className="pointer-events-none absolute left-4 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" />
                  <input
                    value={search}
                    onChange={(event) => setSearch(event.target.value)}
                    placeholder="Search groups, categories, paths, or IDs…"
                    className="h-12 w-full rounded-full border border-slate-300 bg-white pl-11 pr-4 text-sm text-slate-700 shadow-sm transition-colors focus:border-primary-400 focus:ring-2 focus:ring-primary-500/15"
                  />
                </label>
                <div className="flex flex-wrap items-center gap-2 text-xs text-slate-500">
                  <span className="bell-data-pill">{matchingGroupCount} group{matchingGroupCount === 1 ? '' : 's'}</span>
                  <span className="bell-data-pill">{matchingItemCount} categor{matchingItemCount === 1 ? 'y' : 'ies'}</span>
                </div>
              </div>
            </div>

            <div className="grid gap-0 xl:grid-cols-2">
              <div className="border-b border-slate-200/80 px-6 py-5 xl:border-b-0 xl:border-r">
                <div className="mb-4">
                  <div className="text-[11px] font-semibold uppercase tracking-[0.22em] text-emerald-700">Industry group</div>
                  <p className="mt-1 text-sm text-slate-500">
                    Group bridge view that starts from `groups`, then expands into the deeper item hierarchy.
                  </p>
                </div>
                <div className="max-h-[72vh] overflow-auto pr-1">
                  {renderGroupTree()}
                </div>
              </div>

              <div className="px-6 py-5">
                <div className="mb-4">
                  <div className="text-[11px] font-semibold uppercase tracking-[0.22em] text-primary-700">Industry</div>
                  <p className="mt-1 text-sm text-slate-500">
                    Deep hierarchical view of the normalized `items` tree used by the taxonomy mapper.
                  </p>
                </div>
                <div className="max-h-[72vh] overflow-auto pr-1">
                  {visibleRootItems.length > 0 ? (
                    renderItemBranch('0')
                  ) : (
                    <div className="rounded-[1.5rem] border border-dashed border-slate-200 bg-slate-50 px-4 py-10 text-sm text-slate-500">
                      No categories match this search.
                    </div>
                  )}
                </div>
              </div>
            </div>
          </section>

          <div className="xl:sticky xl:top-28 xl:self-start">
            {selectedPanel}
          </div>
        </div>
      ) : null}
    </div>
  );
}
