import {useState} from "react"

function SearchBar({onSearch}) {
    const [input, setInput] = useState("");

    function handleSearchClick(){
        if(input.trim()){
            onSearch(input);
        }
    }

    return (
        <div className="input-group search-bar">
            <input
                type="text"
                className="form-control"
                placeholder="Search ticker"
                aria-label="Ticker"
                aria-describedby="button-addon2"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleSearchClick()}
            />
            <button className="btn btn-outline-secondary" type="button" id="button-addon2" onClick={handleSearchClick}>
                Search
            </button>
        </div>
    );
}

export default SearchBar;