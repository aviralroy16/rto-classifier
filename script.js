// Function to send data to your running local FastAPI backend
async function sendPredictionRequest() {
    const statusEl = document.getElementById('apiStatus');
    const latencyEl = document.getElementById('latency');
    const probEl = document.getElementById('resProb');
    const bandEl = document.getElementById('resBand');
    const actionEl = document.getElementById('resAction');

    // Gather form inputs matching your Pydantic schema
    const payload = {
        order_id: document.getElementById('orderId').value,
        order_amount: parseFloat(document.getElementById('orderAmount').value),
        payment_method: document.getElementById('paymentMethod').value,
        shipping_pincode: parseInt(document.getElementById('pincode').value),
        shipping_address: document.getElementById('shippingAddress').value,
        order_hour: new Date().getHours()
    };

    statusEl.textContent = "Status: Sending...";
    const startTime = performance.now();

    try {
        const response = await fetch('http://127.0.0.1:8000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }

        const data = await response.json();
        const endTime = performance.now();
        const latencyMs = Math.round(endTime - startTime);

        latencyEl.textContent = `Latency: ${latencyMs}ms`;
        statusEl.textContent = "Status: Connected ✓";
        
        // Populate response fields
        probEl.textContent = `${(data.rto_probability * 100).toFixed(1)}%`;
        bandEl.textContent = data.risk_band;
        actionEl.textContent = data.suggested_action;

        // Dynamic coloring based on risk band
        if (data.risk_band === 'HIGH') {
            bandEl.style.color = '#ff5f56';
        } else if (data.risk_band === 'MEDIUM') {
            bandEl.style.color = '#ffbd2e';
        } else {
            bandEl.style.color = '#27c93f';
        }

    } catch (error) {
        statusEl.textContent = "Status: Connection Failed (Check Uvicorn)";
        alert("Could not connect to FastAPI server. Make sure Uvicorn is running on port 8000!");
        console.log(error);
    }
}

// ROI Calculator Logic
function updateROI() {
    const volume = parseInt(document.getElementById('volume').value);
    const shipCost = parseInt(document.getElementById('shipCost').value);

    document.getElementById('volumeVal').textContent = `${volume.toLocaleString()} orders`;
    document.getElementById('costVal').textContent = `₹${shipCost}`;

    // Assuming average 10% baseline RTO reduced down to 7% (3% net reduction)
    const savedOrders = Math.round(volume * 0.03);
    const moneySaved = savedOrders * shipCost * 2; // Two-way shipping saved

    document.getElementById('rtoSaved').textContent = `~${savedOrders.toLocaleString()} orders`;
    document.getElementById('moneySaved').textContent = `₹${moneySaved.toLocaleString()}`;
}

