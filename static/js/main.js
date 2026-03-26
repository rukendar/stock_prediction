// Global chart data store
let currentHistory = [];
let currentPredictions = [];
let currentCompanyName = '';
let currentActiveDays = 30;
let currentTicker = '';

document.addEventListener('DOMContentLoaded', () => {
    // Auth Form Handlers
    const loginForm = document.getElementById('loginForm');
    const signupForm = document.getElementById('signupForm');

    if (loginForm) {
        loginForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const username = loginForm.username.value;
            const password = loginForm.password.value;
            try {
                const res = await fetch('/api/login', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ username, password })
                });
                const data = await res.json();
                if (res.ok) {
                    window.location.href = '/dashboard';
                } else {
                    alert(data.message);
                }
            } catch (err) {
                console.error(err);
            }
        });
    }

    if (signupForm) {
        signupForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const username = signupForm.username.value;
            const email = signupForm.email.value;
            const password = signupForm.password.value;
            try {
                const res = await fetch('/api/signup', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ username, email, password })
                });
                const data = await res.json();
                if (res.ok) {
                    alert('Signup successful! Please login.');
                    window.location.href = '/login';
                } else {
                    alert(data.message);
                }
            } catch (err) {
                console.error(err);
            }
        });
    }

    // Dashboard stock search
    const stockSearch = document.getElementById('stockSearch');
    if (stockSearch) {
        stockSearch.addEventListener('input', (e) => {
            const term = e.target.value.toLowerCase();
            const items = document.querySelectorAll('#stockList li');
            items.forEach(item => {
                const text = item.innerText.toLowerCase();
                item.style.display = text.includes(term) ? 'flex' : 'none';
            });
        });
    }
});

// Custom mapped timeframe intervals per user request
const TF_MAP = {
    1:    { period: '1d',  interval: '5m',  label: '5-Min',    live: true },
    7:    { period: '7d',  interval: '30m', label: '30-Min',   live: true },  // 1 week -> 30 mins
    30:   { period: '1mo', interval: '60m', label: '1-Hour',   live: true },  // 1 month -> ~1 hour (yfinance 2h workaround)
    90:   { period: '3mo', interval: '1d',  label: 'Daily',    live: true },  // 3 months -> 1 day
    180:  { period: '6mo', interval: '1d',  label: 'Daily',    live: true },  // 6 months -> 1 day
    365:  { period: '1y',  interval: '1d',  label: 'Daily',    live: true },  // yfinance max interval for 1y is 1d/1wk, 1d gives the approx 3-day detail
    1825: { period: '5y',  interval: '1mo', label: 'Monthly',  live: true },
    null: { period: 'max', interval: '1mo', label: 'Monthly',  live: true }
};

async function setTimeframe(days) {
    currentActiveDays = days;
    const tf = TF_MAP[days] || TF_MAP[30];

    // Update button active states
    document.querySelectorAll('.tf-btn').forEach(btn => btn.classList.remove('active'));
    event.target.classList.add('active');

    const chartDiv = document.getElementById('stockChart');

    if (tf.live) {
        // Fetch live data at the correct interval from yfinance
        Plotly.purge('stockChart');
        chartDiv.innerHTML = `<div style="text-align:center;padding:60px;color:var(--text-dim)">⏳ Loading ${tf.label} candles...</div>`;
        try {
            const url = days === 1
                ? `/api/intraday/${currentTicker}`
                : `/api/ohlc/${encodeURIComponent(currentTicker)}?period=${tf.period}&interval=${tf.interval}`;
            const res  = await fetch(url);
            const data = await res.json();
            if (data.error) {
                chartDiv.innerHTML = `<div style="text-align:center;padding:60px;color:#f85149">⚠️ ${data.error}</div>`;
                return;
            }
            const candles = data.candles;
            const dateKey  = data.date_key || 'Datetime';

            // Update rest OHLC to last candle
            const last = candles[candles.length - 1];
            if (last) {
                document.getElementById('ohlcOpen').innerText  = `₹${parseFloat(last.Open).toFixed(2)}`;
                document.getElementById('ohlcHigh').innerText  = `₹${parseFloat(last.High).toFixed(2)}`;
                document.getElementById('ohlcLow').innerText   = `₹${parseFloat(last.Low).toFixed(2)}`;
                document.getElementById('ohlcClose').innerText = `₹${parseFloat(last.Close).toFixed(2)}`;
            }

            renderIntradayChart(candles, currentCompanyName, tf.label);
        } catch (err) { console.error(err); }
        return;
    }

    // For 1M/3M/6M: use stored daily CSV data (no extra API call)
    let filtered = currentHistory;
    if (days !== null) {
        const cutoff = new Date();
        cutoff.setDate(cutoff.getDate() - days);
        const cutoffStr = cutoff.toISOString().split('T')[0];
        filtered = currentHistory.filter(h => h.Date >= cutoffStr);
    }
    Plotly.purge('stockChart');
    renderChart(filtered, currentPredictions, currentCompanyName);
}

function renderIntradayChart(candles, title, intervalLabel) {
    // Always clear the div first before Plotly renders into it
    const chartDiv = document.getElementById('stockChart');
    chartDiv.innerHTML = '';
    const label = intervalLabel || '5-Min';

    const trace = {
        x: candles.map(c => c.Date || c.Datetime),  // both intraday and /api/ohlc use .Date
        open: candles.map(c => c.Open),
        high: candles.map(c => c.High),
        low: candles.map(c => c.Low),
        close: candles.map(c => c.Close),
        type: 'candlestick',
        name: `${label} Candles`,
        increasing: { line: { color: '#00d09c', width: 1.5 }, fillcolor: '#00d09c' },
        decreasing: { line: { color: '#f85149', width: 1.5 }, fillcolor: '#f85149' },
        hoverinfo: 'none' // Disable standard tooltip box to match Groww (we use header sync)
    };

    const layout = {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#e6edf3' },
        margin: { t: 10, r: 0, l: 0, b: 0 },
        xaxis: {
            type: 'date',
            rangebreaks: [
                { bounds: ['sat', 'mon'] }, // Hide weekends
                { bounds: [15.5, 9.25], pattern: 'hour' } // Hide non-trading hours (15:30 to 09:15)
            ],
            showgrid: false,
            zeroline: false,
            showline: false,
            showticklabels: false,
            rangeslider: { visible: false },
            showspikes: true,
            spikemode: 'across',
            spikedash: 'solid',
            spikecolor: '#30363d',
            spikethickness: 1
        },
        yaxis: { 
            showgrid: false,
            zeroline: false,
            showline: false,
            showticklabels: false
        },
        showlegend: false,
        hovermode: 'x unified'
    };

    Plotly.newPlot('stockChart', [trace], layout, {
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToRemove: ['lasso2d', 'select2d', 'autoScale2d'],
        displaylogo: false
    }).then(() => {
        attachOHLCHoverSync('stockChart', candles, 'Datetime');
    });
}

/**
 * Attaches Plotly hover events to update the OHLC header row live.
 * @param {string} divId - Plotly chart div ID
 * @param {Array} candles - Array of {Open, High, Low, Close, Date/Datetime} objects
 * @param {string} dateKey - Key for the date/datetime field ('Date' or 'Datetime')
 */
function attachOHLCHoverSync(divId, candles, dateKey) {
    const chartDiv = document.getElementById(divId);
    const hoverDateEl = document.getElementById('hoverDate');
    
    // Save the "resting" values — last candle
    const lastCandle = candles[candles.length - 1];

    function updateOHLC(o, h, l, c) {
        document.getElementById('ohlcOpen').innerText  = `₹${parseFloat(o).toFixed(2)}`;
        document.getElementById('ohlcHigh').innerText  = `₹${parseFloat(h).toFixed(2)}`;
        document.getElementById('ohlcLow').innerText   = `₹${parseFloat(l).toFixed(2)}`;
        document.getElementById('ohlcClose').innerText = `₹${parseFloat(c).toFixed(2)}`;
    }

    function formatHoverDate(xVal) {
        try {
            const d = new Date(xVal);
            if (isNaN(d)) return xVal; // fallback for 'YYYY-MM-DD HH:MM' strings
            const opts = dateKey === 'Datetime'
                ? { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit', hour12: false }
                : { weekday: 'short', day: 'numeric', month: 'short', year: 'numeric' };
            return d.toLocaleString('en-IN', opts);
        } catch(e) { return xVal; }
    }

    chartDiv.on('plotly_hover', (eventData) => {
        try {
            const pt = eventData.points[0];
            if (pt.open !== undefined) {
                updateOHLC(pt.open, pt.high, pt.low, pt.close);
            }
            // Move the absolute date label along the X-axis to track the crosshair
            if (hoverDateEl && eventData.event) {
                // Plotly eventData.event contains the browser mouse event
                // We use offsetX relative to the innermost plotly layer
                const bounding = chartDiv.getBoundingClientRect();
                const mouseX = eventData.event.clientX - bounding.left;
                
                hoverDateEl.style.left = `${mouseX}px`;
                hoverDateEl.innerText = formatHoverDate(pt.x);
                hoverDateEl.classList.add('visible');
            }
        } catch(e) {}
    });

    chartDiv.on('plotly_unhover', () => {
        if (lastCandle) {
            updateOHLC(lastCandle.Open, lastCandle.High, lastCandle.Low, lastCandle.Close);
        }
        if (hoverDateEl) {
            hoverDateEl.classList.remove('visible');
        }
    });
}

async function loadStock(ticker) {
    // Re-show AI sections (may have been hidden by loadIndex)
    document.querySelector('.forecast-section').style.display = '';
    document.querySelector('.validation-section').style.display = '';
    document.querySelector('.ai-badge').style.opacity = '1';

    document.getElementById('selectionHint').classList.add('hidden');
    document.getElementById('stockDetails').classList.remove('hidden');
    
    document.getElementById('displayTicker').innerText = 'Loading...';
    document.getElementById('displayCompany').innerText = '';
    
    try {
        const res = await fetch(`/api/predict/${ticker}`);
        const data = await res.json();
        
        if (data.error) {
            alert(data.error);
            return;
        }

        // Store globally for timeframe switching
        currentTicker = ticker;
        currentHistory = data.history;
        currentPredictions = data.predictions;
        currentCompanyName = data.company_name || data.ticker;

        document.getElementById('displayCompany').innerText = currentCompanyName;
        document.getElementById('displayTicker').innerText = data.ticker;
        document.getElementById('displayPrice').innerText = `₹${data.current_price.toLocaleString('en-IN', {minimumFractionDigits: 2})}`;

        // Compute day change from history (last vs prev close)
        const changeEl = document.getElementById('displayChange');
        if (currentHistory.length >= 2) {
            const prevClose = currentHistory[currentHistory.length - 2].Close;
            const currClose = data.current_price;
            const chg = currClose - prevClose;
            const chgPct = (chg / prevClose) * 100;
            const sign = chg >= 0 ? '+' : '';
            changeEl.innerText = `${sign}${chg.toFixed(2)} (${sign}${chgPct.toFixed(2)}%)`;
            changeEl.className = `groww-change ${chg >= 0 ? 'up' : 'down'}`;
        }

        // Fill OHLC from latest candle in history
        const latestCandle = currentHistory[currentHistory.length - 1];
        if (latestCandle) {
            document.getElementById('ohlcOpen').innerText = `₹${latestCandle.Open.toFixed(2)}`;
            document.getElementById('ohlcHigh').innerText = `₹${latestCandle.High.toFixed(2)}`;
            document.getElementById('ohlcLow').innerText = `₹${latestCandle.Low.toFixed(2)}`;
            document.getElementById('ohlcClose').innerText = `₹${latestCandle.Close.toFixed(2)}`;
        }

        const pred1 = data.predictions[0];
        document.getElementById('predictionValue').innerText = pred1 ? `₹${pred1.toLocaleString('en-IN', {minimumFractionDigits: 2})}` : 'N/A';

        // Reset to 1M view and mark 1M button active
        document.querySelectorAll('.tf-btn').forEach(btn => btn.classList.remove('active'));
        const oneMBtn = document.querySelector('.tf-btn[data-days="30"]');
        if (oneMBtn) oneMBtn.classList.add('active');
        
        // Render default 1M chart view
        const cutoff = new Date();
        cutoff.setDate(cutoff.getDate() - 30);
        const cutoffStr = cutoff.toISOString().split('T')[0];
        const filtered = currentHistory.filter(h => h.Date >= cutoffStr);
        Plotly.purge('stockChart');
        renderChart(filtered, currentPredictions, currentCompanyName);
        
        renderForecastCards(data.current_price, data.predictions);
        renderValidationTable(data.validation);
    } catch (err) {
        console.error(err);
    }
}

function renderValidationTable(validation) {
    const tbody = document.getElementById('validationBody');
    tbody.innerHTML = '';
    
    if (!validation || validation.length === 0) {
        tbody.innerHTML = '<tr><td colspan="5">No validation data available</td></tr>';
        return;
    }

    validation.forEach(v => {
        const row = document.createElement('tr');
        const dirIcon = v.direction_correct
            ? '<span style="color:#00d09c;font-size:1.1rem;">↑ Correct</span>'
            : '<span style="color:#f85149;font-size:1.1rem;">↓ Wrong</span>';
        row.innerHTML = `
            <td>${v.date}</td>
            <td>₹${v.actual.toLocaleString('en-IN', {minimumFractionDigits: 2})}</td>
            <td>₹${v.predicted.toLocaleString('en-IN', {minimumFractionDigits: 2})}</td>
            <td class="variance-cell ${v.variance_pct < 2 ? 'good' : ''}">₹${v.variance_abs.toFixed(2)} (${v.variance_pct.toFixed(2)}%)</td>
            <td>${dirIcon}</td>
        `;
        tbody.appendChild(row);
    });
}

function renderForecastCards(currentPrice, predictions) {
    const grid = document.getElementById('forecastGrid');
    grid.innerHTML = '';
    
    const days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];
    const today = new Date();
    
    predictions.forEach((pred, i) => {
        if (!pred) return;
        
        const forecastDate = new Date();
        forecastDate.setDate(today.getDate() + i + 1);
        const dayName = forecastDate.toLocaleDateString('en-US', { weekday: 'long' });
        const dateStr = forecastDate.toLocaleDateString('en-US', { day: 'numeric', month: 'short' });
        
        const delta = pred - currentPrice;
        const deltaPct = (delta / currentPrice) * 100;
        const deltaClass = delta >= 0 ? 'up' : 'down';
        const deltaSym = delta >= 0 ? '+' : '';

        const card = document.createElement('div');
        card.className = 'forecast-card';
        card.innerHTML = `
            <div class="day">${dayName} (${dateStr})</div>
            <div class="price">₹${pred.toLocaleString('en-IN', {minimumFractionDigits: 2})}</div>
            <div class="delta ${deltaClass}">${deltaSym}${delta.toFixed(2)} (${deltaSym}${deltaPct.toFixed(2)}%)</div>
        `;
        grid.appendChild(card);
    });
}

function renderChart(history, predictions, ticker) {
    const dates = history.map(h => h.Date);
    
    const trace1 = {
        x: dates,
        open: history.map(h => h.Open),
        high: history.map(h => h.High),
        low: history.map(h => h.Low),
        close: history.map(h => h.Close),
        type: 'candlestick',
        name: 'Price Action',
        increasing: { line: { color: '#00d09c', width: 1.5 }, fillcolor: '#00d09c' },
        decreasing: { line: { color: '#f85149', width: 1.5 }, fillcolor: '#f85149' },
        hoverinfo: 'none' // Disable standard tooltip box to match Groww
    };

    const traces = [trace1];

    if (predictions && predictions.length > 0) {
        let forecastX = [dates[dates.length - 1]];
        let forecastY = [history[history.length - 1].Close];
        let forecastText = [`Last Close: ₹${history[history.length - 1].Close.toFixed(2)}`];
        
        const lastDate = new Date(dates[dates.length - 1]);
        
        predictions.forEach((pred, i) => {
            if (pred) {
                const nextDate = new Date(lastDate);
                nextDate.setDate(lastDate.getDate() + i + 1);
                forecastX.push(nextDate.toISOString().split('T')[0]);
                forecastY.push(pred);
                forecastText.push(`Day +${i+1} Forecast: ₹${pred.toFixed(2)}`);
            }
        });
        
        const trace2 = {
            x: forecastX,
            y: forecastY,
            text: forecastText,
            type: 'scatter',
            mode: 'lines+markers',
            name: 'AI 5-Day Forecast',
            line: { color: '#00d09c', width: 2, dash: 'dot' },
            marker: { size: 7, color: '#00d09c', symbol: 'circle' },
            hovertemplate: '%{text}<extra></extra>',
            hoverlabel: { bgcolor: '#161b22', bordercolor: '#00d09c', font: { color: '#e6edf3', size: 12 } }
        };
        traces.push(trace2);
    }

    const layout = {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#e6edf3' },
        margin: { t: 10, r: 0, l: 0, b: 0 },
        xaxis: { 
            type: 'date',
            rangebreaks: [
                { bounds: ['sat', 'mon'] } // Hide weekends for daily charts
            ],
            showgrid: false,
            zeroline: false,
            showline: false,
            showticklabels: false,
            rangeslider: { visible: false },
            showspikes: true,
            spikemode: 'across',
            spikedash: 'solid',
            spikecolor: '#30363d',
            spikethickness: 1
        },
        yaxis: { 
            showgrid: false,
            zeroline: false,
            showline: false,
            showticklabels: false
        },
        hovermode: 'x',
        showlegend: true,
        legend: { orientation: 'h', y: 1.06, font: { size: 11 } }
    };

    Plotly.newPlot('stockChart', traces, layout, {
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToRemove: ['lasso2d', 'select2d', 'autoScale2d'],
        displaylogo: false
    }).then(() => {
        attachOHLCHoverSync('stockChart', history, 'Date');
    });
}
