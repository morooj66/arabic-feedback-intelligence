/**
 * Explicit selection-scope control. Never lets a paginated checkbox
 * silently select unseen records — "select all filtered results" always
 * states the exact count it means.
 */
export function SelectionBar({
  selectedCount,
  pageCount,
  totalCount,
  allFilteredSelected,
  onSelectPage,
  onSelectAllFiltered,
  onClear,
}: {
  selectedCount: number;
  pageCount: number;
  totalCount: number;
  allFilteredSelected: boolean;
  onSelectPage: () => void;
  onSelectAllFiltered: () => void;
  onClear: () => void;
}) {
  return (
    <div className="selection-bar">
      <span className="selection-bar-count">
        {selectedCount > 0 ? `${selectedCount} تعليقًا محددًا` : "لم يتم تحديد عناصر"}
      </span>
      <div className="selection-bar-actions">
        <button type="button" className="linkish" onClick={onSelectPage} disabled={pageCount === 0}>
          تحديد الصفحة الحالية ({pageCount})
        </button>
        {totalCount > pageCount ? (
          <button type="button" className="linkish" onClick={onSelectAllFiltered} disabled={allFilteredSelected}>
            تحديد جميع نتائج الفلترة ({totalCount})
          </button>
        ) : null}
        {selectedCount > 0 ? (
          <button type="button" className="linkish" onClick={onClear}>
            إلغاء التحديد
          </button>
        ) : null}
      </div>
    </div>
  );
}
