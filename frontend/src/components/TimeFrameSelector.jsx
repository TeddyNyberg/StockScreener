function TimeFrameSelector({timeRange, setTimeRange, loading}){
    const timeButtons = ["1D", "5D", "1M", "3M", "6M", "YTD", "1Y", "5Y", "MAX"];
    return (
        <div className="my-3">
            {timeButtons.map(time => (
                <button
                    key={time}
                    disabled={loading}
                    style={{margin: "2px"}}
                    className={`btn btn-sm ${timeRange === time ? 'btn-primary' : 'btn-outline-secondary'}`}
                    onClick={() => setTimeRange(time)}>
                    {time}
                </button>
            ))}
            {loading && <span className="spinner-border spinner-border-sm ms-2 text-secondary"></span>}
        </div>
    );
}export default TimeFrameSelector;