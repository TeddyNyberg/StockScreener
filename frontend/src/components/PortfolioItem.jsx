import {getColor, formatCurrency, formatPercent} from '../utils/formatting';

function PortfolioItem({item}) {
    return (
        <tr>
            <td className="fw-bold">{item.ticker}</td>

            <td className="text-end">{item.shares}</td>

            <td className="text-end">
                ${item.avg ? item.avg.toFixed(2) : "0.00"}
            </td>

            <td className="text-end">
                ${item.price ? item.price.toFixed(2) : "0.00"}
            </td>

            <td className="text-end fw-bold">
                ${item.market_value ? item.market_value.toFixed(2) : "0.00"}
            </td>

            <td className={`text-end ${getColor(item.pl)}`}>
                {item.pl > 0 ? "+" : ""}
                {item.pl ? item.pl.toFixed(2) : "0.00"}
            </td>

            <td className={`text-end ${getColor(item.pct)}`}>
                {item.pct > 0 ? "+" : ""}
                {item.pct ? item.pct.toFixed(2) : "0.00"}%
            </td>
        </tr>
    )

}

export default PortfolioItem;