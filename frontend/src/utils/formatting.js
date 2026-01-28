export function getColor(val) {
    if (val > 0) return "text-success";
    if (val < 0) return "text-danger";
    return "text-muted";
}

export function formatCurrency(val) {
    return val ? `$${val.toFixed(2)}` : "---";
}

export function formatPercent(val) {
    return val ? `${val.toFixed(2)}%` : "-";
}