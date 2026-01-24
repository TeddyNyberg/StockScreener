function SearchBar() {
    return (
        <div className="input-group search-bar">
            <input
                type="text"
                className="form-control"
                placeholder="Search ticker"
                aria-label="Ticker"
                aria-describedby="button-addon2"
            />
            <button className="btn btn-outline-secondary" type="button" id="button-addon2">
                Search
            </button>
        </div>
    );
}

export default SearchBar;