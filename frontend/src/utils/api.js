const BASE_URL = "http://localhost:8000";

export async function apiRequest(endpoint, options = {}) {
    const token = sessionStorage.getItem("token");

    const headers = {
        "Content-Type": "application/json",
        ...(token && { "Authorization": `Bearer ${token}` }),
        ...options.headers,
    };

    const config = {
        method: "GET",
        ...options,    // If options contains a 'method', it will overwrite 'GET'
        headers,
    };

    const response = await fetch(`${BASE_URL}${endpoint}`, config);
    if (response.status === 401) {
        sessionStorage.removeItem("token");
        window.location.href = "/";
    }

    if (!response.ok) {
        const error = await response.json().catch(() => ({}));
        throw new Error(error.detail || `API Error: ${response.status}`);
    }

    return response.json();
}