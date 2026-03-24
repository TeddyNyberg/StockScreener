

export function processChartData(apiResponse, tickers) {

    if (!apiResponse || tickers.length === 0) {
        return [];
    }

    const tickerDateSets = tickers.map(ticker => {
        const dates = apiResponse[ticker]?.map(row => row.Date) || [];
        return new Set(dates);
    });

    const firstTickerDates = Array.from(tickerDateSets[0] || []);
    const commonDates = firstTickerDates.filter(date =>
        tickerDateSets.every(set => set.has(date))
    );

    const sortedDates = commonDates.sort((a, b) => new Date(a) - new Date(b));

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