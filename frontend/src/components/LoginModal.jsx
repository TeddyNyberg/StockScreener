import React, { useState } from 'react';
import axios from 'axios';
import {useAuth} from "../context/AuthContext.jsx";


function LoginModal() {
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState('');


    const { showLogin, setShowLogin, login } = useAuth();

    if (!showLogin) return null;

    async function handleLogin(e) {
        e.preventDefault();
        setError('');

        const formData = new URLSearchParams();
        formData.append("username", username);
        formData.append("password", password);

        try {
            const response = await axios.post('http://localhost:8000/token',
                formData,
                { headers: {
                        "Content-Type": "application/x-www-form-urlencoded",
                    }
                });
            login(username, response.data.access_token);
        } catch (err) {
            console.error(err);
            setError("Invalid credentials");
        }
    }

    return (

        <div className="modal show d-block" tabIndex="-1" style={{ backgroundColor: 'rgba(0,0,0,0.7)' }}>

            <div className="modal-dialog modal-dialog-centered">
                <div className="modal-content bg-dark text-white" style={{ border: '1px solid #444', boxShadow: '0 0 20px rgba(0,0,0,0.5)' }}>

                    <div className="modal-header border-secondary">
                        <h5 className="modal-title">Sign In</h5>
                        <button type="button" className="btn-close btn-close-white" onClick={() => setShowLogin(false)}></button>
                    </div>

                    <div className="modal-body">
                        {error && <div className="alert alert-danger">{error}</div>}
                        <form onSubmit={handleLogin}>
                            <div className="mb-3">
                                <label className="form-label">Username</label>
                                <input
                                    type="text" className="form-control bg-secondary text-white border-0"
                                    value={username} onChange={(e) => setUsername(e.target.value)}
                                    autoFocus
                                />
                            </div>
                            <div className="mb-3">
                                <label className="form-label">Password</label>
                                <input
                                    type="password" className="form-control bg-secondary text-white border-0"
                                    value={password} onChange={(e) => setPassword(e.target.value)}
                                />
                            </div>
                            <button type="submit" className="btn btn-primary w-100">Login</button>
                        </form>
                    </div>
                </div>
            </div>
        </div>
    );
}

export default LoginModal;