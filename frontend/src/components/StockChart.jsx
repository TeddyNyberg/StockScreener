import React, { useMemo } from 'react';
import {
    AreaChart, Area, XAxis, YAxis, CartesianGrid,
    Tooltip, ResponsiveContainer, Legend
} from 'recharts';
import { processChartData } from "../utils/chartUtils";
import "./StockChart.css";

const COLORS = ["#8884d8", "#82ca9d", "#ffc658", "#ff7300", "#d0ed57", "#a4de6c"];

function StockChart({ apiData, tickers }) {


    const isMulti = tickers.length > 1;

    const chartData = useMemo(() => {
        return processChartData(apiData, tickers);
    }, [apiData, tickers]);

    const CustomTooltip = ({active, payload, label}) => {
        if (active && payload && payload.length) {
            return (
                <div className="custom-tooltip"
                     style={{backgroundColor: '#333', padding: '10px', borderRadius: '5px', border: '1px solid #555'}}>
                    <p style={{color: '#fff', marginBottom: '5px'}}>{label}</p>
                    {payload.map((entry, index) => (
                        <div key={index} style={{color: entry.color}}>
                            {entry.name}: {entry.value?.toFixed(2)}{isMulti ? '%' : ''}
                        </div>
                    ))}
                </div>
            );
        }
        return null;
    };

    const allValues = chartData.flatMap(d =>
        tickers.map(ticker => d[ticker]).filter(val => val !== undefined)
    );

    const maxValue = Math.max(...allValues);
    const minValue = Math.min(...allValues);

    const zeroOffset = maxValue > 0 && minValue < 0
        ? (maxValue / (maxValue - minValue)) * 100
        : (maxValue <= 0 ? 0 : 100);

    return (
        <div className="mx-auto full-figure" style={{width: '100%', height: 400}}>
            <h2 className="text-center">
                {tickers.join(" vs ")} {isMulti ? "(% Return)" : "(Price)"}
            </h2>

            <ResponsiveContainer>
                <AreaChart data={chartData} margin={{top: 10, right: 30, left: 0, bottom: 0}}>
                    <defs>
                        {tickers.map((ticker, index) => {
                            const color = COLORS[index % COLORS.length];
                            return (
                                <>
                                    <linearGradient key={ticker} id={`color${ticker}`} x1="0" x2="0" y1="0" y2="1">
                                        <stop offset="0%" stopColor={color} stopOpacity={0.8}/>
                                        <stop offset={`${zeroOffset}%`} stopColor={color} stopOpacity={0.1}/>

                                        <stop offset={`${zeroOffset}%`} stopColor="white" stopOpacity={0}/>

                                        <stop offset={`${zeroOffset}%`} stopColor={color} stopOpacity={0.1}/>
                                        <stop offset="100%" stopColor={color} stopOpacity={0.8}/>
                                    </linearGradient>
                                </>
                            );
                        })}
                    </defs>

                    <CartesianGrid strokeDasharray="3 3" opacity={0.1}/>

                    <XAxis
                        dataKey="date"
                        tickFormatter={(str) => {
                            const d = new Date(str);
                            return `${d.getMonth() + 1}/${d.getDate()}`;
                        }}
                    />

                    <YAxis
                        domain={['auto', 'auto']}
                        tickFormatter={(number) => isMulti ? `${number}%` : `$${number}`}
                    />

                    <Tooltip content={<CustomTooltip/>}/>
                    <Legend/>

                    {tickers.map((ticker, index) => {
                        const color = COLORS[index % COLORS.length];
                        return (
                            <Area
                                key={ticker}
                                type="monotone"
                                dataKey={ticker}
                                stroke={color}
                                fillOpacity={1}
                                fill={`url(#color${ticker})`}
                                name={ticker}
                            />
                        );
                    })}
                </AreaChart>
            </ResponsiveContainer>
        </div>
    );
};

export default StockChart;