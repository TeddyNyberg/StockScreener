
import React, {useMemo} from 'react';
import ReactECharts from 'echarts-for-react';
import * as echarts from 'echarts';
import { processChartData } from "../utils/chartUtils";
import "./StockChart.css";

const COLORS = ["#8884d8", "#82ca9d", "#ffc658", "#ff7300", "#d0ed57", "#a4de6c"];

function StockChart({ apiData, tickers }) {
    const isMulti = tickers.length > 1;

    // Process data using your existing utility
    const chartData = useMemo(() => {
        return processChartData(apiData, tickers);
    }, [apiData, tickers]);

    const option = useMemo(() => {
        const xLabels = chartData.map(d => d.date);
        const series = tickers.map((ticker, index) => {
            const color = COLORS[index % COLORS.length];

            const dataValues = chartData.map(d => d[ticker]);

            return {
                name: ticker,
                type: 'line',
                data: dataValues,
                smooth: false,
                showSymbol: false,
                itemStyle: {
                    color: color
                },
                lineStyle: {
                    width: 2
                },
                // Only add Area gradient if it is a single ticker
                areaStyle: !isMulti ? {
                    opacity: 0.8,
                    color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                        { offset: 0.05, color: color },
                        { offset: 0.95, color: 'rgba(255, 255, 255, 0)' }
                    ])
                } : undefined
            };
        });

        return {
            title: {
                text: `${tickers.join(" vs ")} ${isMulti ? "(% Return)" : "(Price)"}`,
                left: 'center',
                textStyle: { fontSize: 16 }
            },
            tooltip: {
                trigger: 'axis',
                axisPointer: {
                    type: 'cross',
                    crossStyle: {
                        color: '#999',
                        width: 1,
                        type: 'dashed'
                    },
                    label: {
                        backgroundColor: '#666'
                    }
                },

                backgroundColor: '#333',
                borderColor: '#555',
                textStyle: {
                    color: '#fff'
                },
                // formatter for tooltip
                formatter: function (params) {
                    let tooltipContent = `<div style="margin-bottom: 5px; font-weight:bold;">${params[0].axisValueLabel}</div>`;

                    params.forEach(param => {
                        const valueDisplay = param.value !== undefined
                            ? (typeof param.value === 'number' ? param.value.toFixed(2) : param.value)
                            : 'N/A';

                        const suffix = isMulti ? '%' : '';

                        const marker = `<span style="display:inline-block;margin-right:5px;border-radius:10px;width:10px;height:10px;background-color:${param.color};"></span>`;

                        tooltipContent += `<div>${marker} ${param.seriesName}: ${valueDisplay}${suffix}</div>`;
                    });
                    return tooltipContent;
                }
            },
            legend: {
                data: tickers,
                bottom: 0
            },
            grid: {
                top: 50,
                left: 20,
                right: 30,
                bottom: 30,
                containLabel: true,
                show: true,
                borderColor: 'transparent' // Hide border

            },
            xAxis: {
                type: 'category',
                data: xLabels,
                boundaryGap: false,
                axisLine: { show: true },
                axisTick: { show: false },
                splitLine: {
                    show: true,
                    lineStyle: {
                        type: 'dashed',
                        opacity: 0.5
                    }
                },
                axisLabel: {
                    formatter: (value) => {
                        const d = new Date(value);
                        return `${d.getMonth() + 1}/${d.getDate()}`;
                    },
                    color: '#666'
                }
            },
            yAxis: {
                type: 'value',
                scale: true,
                splitLine: {
                    show: true,
                    lineStyle: {
                        type: 'dashed',
                        opacity: 0.5
                    }
                },
                axisLabel: {
                    formatter: (value) => isMulti ? `${value}%` : `${value}`,
                    color: '#666'
                },
                axisLine: { show: true },
            },
            series: series
        };
    }, [chartData, tickers, isMulti]);

    return (
        <div className="mx-auto full-figure" style={{ width: '100%', height: 400 }}>
            <ReactECharts
                option={option}
                style={{ height: '100%', width: '100%' }}
                notMerge={true}
            />
        </div>
    );
}

export default StockChart;