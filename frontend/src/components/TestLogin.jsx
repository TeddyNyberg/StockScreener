

import React, { useState } from 'react';
import axios from 'axios';
import {useAuth} from "../context/AuthContext.jsx";

function TestLogin(){

    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState('');


    const { showLogin, login, setShowLogin } = useAuth();

    return (

        <div className="modal show d-block" tabIndex="-1" style={{ backgroundColor: 'rgba(0,0,0,0.7)' }}>

            <div className="modal-dialog modal-dialog-centered">
                <div className="modal-content bg-dark text-white" style={{ border: '1px solid #444', boxShadow: '0 0 20px rgba(0,0,0,0.5)' }}>

                    <div className="modal-header border-secondary">
                        <h5 className="modal-title">Sign In</h5>
                        <button type="button" className="btn-close btn-close-white" onClick={() => setShowLogin(false)}></button>
                    </div>
                </div>
            </div>
        </div>
    );
}export default TestLogin;