

export function processChartData(apiResponse, tickers) {

    if (!apiResponse || tickers.length === 0) {
        return [];
    }

    const allDates = new Set();

    Object.values(apiResponse).forEach(function(rows) {
        rows.forEach(function(row) {
            allDates.add(row.Date);
        });
    });

    const sortedDates = Array.from(allDates).sort();

    const shouldNormalize = tickers.length > 1;
    const startPrices = {};


    return sortedDates.map(function(date) {
        const row = { date: date };

        tickers.forEach(function(ticker) {
            const dayData = apiResponse[ticker]?.find(function(d) {
                return d.Date === date;
            });

            if (dayData) {
                const price = dayData.Close;

                if (startPrices[ticker] === undefined) {
                    startPrices[ticker] = price;
                }

                if (shouldNormalize && startPrices[ticker]) {
                    const pctChange = ((price - startPrices[ticker]) / startPrices[ticker]) * 100;
                    row[ticker] = pctChange;
                    row[`${ticker}_raw`] = price;
                } else {
                    row[ticker] = price;
                    row[`${ticker}_raw`] = price;
                }
            } else {
                row[ticker] = null;
            }
        });
        return row;
      });
}