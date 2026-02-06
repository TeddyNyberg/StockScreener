

function ViewSelector({currentView, handleViewChange}){
    const VIEWS = [
        {label: "General Info", value: ""},
        {label: "Financials", value: "financials"},
        {label: "Balance Sheet", value: "balance_sheet"},
        {label: "Focused View", value: "my_chart"},
    ];

    return (<div className="btn-group mb-4 shadow-sm" role="group">
        {VIEWS.map((view) => (
            <button
                key={view.label}
                type="button"
                className={`btn ${currentView === view.value ? "btn-primary" : "btn-outline-primary"}`}
                onClick={() => handleViewChange(view.value)}
            >
                {view.label}
            </button>
        ))}
    </div>);
}export default ViewSelector;