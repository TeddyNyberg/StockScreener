import {createContext, useContext, useEffect, useState} from "react";


const AuthContext = createContext();

export function AuthProvider({children}){
    const [user, setUser] = useState(null);
    const [showLogin, setShowLogin] = useState(false);

    useEffect(() =>{
        const storedUser = sessionStorage.getItem("username");
        const token = sessionStorage.getItem("token");

        if (token && storedUser) {
            setUser(storedUser);
        }

    }, []);

    function login(username, token){
        sessionStorage.setItem("username", username);
        sessionStorage.setItem("token", token);

        setUser(username);
        setShowLogin(false);
    }

    function logout(){
        sessionStorage.clear();
        setUser(null);
    }

    return (
        <AuthContext.Provider value={{ user, showLogin, setShowLogin, login, logout}}>
            {children}
        </AuthContext.Provider>
    );
}

export function useAuth(){
    return useContext(AuthContext);
}