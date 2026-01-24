import React from 'react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import "./StockChart.css"

const data = [
  { date: '2025-01-01', price: 400.50 },
  { date: '2025-01-02', price: 405.10 },
  { date: '2025-01-03', price: 402.00 },
  { date: '2025-01-04', price: 408.90 },
  { date: '2025-01-05', price: 410.50 },
  { date: '2025-01-06', price: 415.20 },
];

const name = "SPY"

function StockChart() {
  return (
    <div className="mx-auto full-figure">
      <h1 className="text-center">{name}</h1>
      <ResponsiveContainer className="chart">
        <AreaChart data={data}>
          <defs>
            <linearGradient id="colorPrice" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#8884d8" stopOpacity={0.8}/>
              <stop offset="95%" stopColor="#8884d8" stopOpacity={0}/>
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="date" />
          <YAxis domain={['auto', 'auto']} />
          <Tooltip
            contentStyle={{ backgroundColor: '#333', color: '#fff', borderRadius: '5px' }}
            itemStyle={{ color: '#fff' }}
          />
          <Area type="monotone" dataKey="price" stroke="#8884d8" fillOpacity={1} fill="url(#colorPrice)" />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
};

export default StockChart;