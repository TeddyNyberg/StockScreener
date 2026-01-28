import { getColor, formatCurrency, formatPercent } from '../utils/formatting';

function WatchlistItem ({item}){
    return (
        <li className="list-group-item d-flex justify-content-between align-items-center">
            <div>
                <h5 className="mb-0">{item.ticker}</h5>
            </div>
            <div className="text-end">
                <div className="fw-bold">
                    {formatCurrency(item.price)}
                </div>
                <div className={`small ${getColor(item.change)}`}>
                    {item.change > 0 ? "+" : ""}
                    {item.change ? item.change.toFixed(2) : "-"}
                    <span className="ms-1">
                        {formatPercent(item.change_percent)}
                    </span>
                </div>
            </div>
            <button className="btn btn-sm btn-outline-danger ms-3">X</button>
        </li>
    )

}
export default WatchlistItem;