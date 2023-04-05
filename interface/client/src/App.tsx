import React, { useState, useRef, useEffect, ChangeEvent, FormEvent } from 'react';
import axios from 'axios';
import './App.css';

interface Data {
  id: number;
  prompt: string;
  completion: string;
}

const App: React.FC = () => {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<Data[]>([]);
  const [selectedData, setSelectedData] = useState<Data | null>(null);
  const [fields, setFields] = useState<Data>({} as Data);
  const searchContainerRef = useRef<HTMLDivElement>(null);
  const [updating, setUpdating] = useState(false);

  useEffect(() => {
    if (selectedData || results.length > 0) {
      searchContainerRef.current?.classList.add('top');
    } else {
      searchContainerRef.current?.classList.remove('top');
    }
  }, [selectedData, results]);

  const fetchData = async () => {
    try {
      const response = await axios.get<Data[]>(`http://127.0.0.1:5000/api/search?query=${query}`);
      setResults(response.data);
    } catch (error) {
      console.error('Error fetching data:', error);
    }
  };

  const updateData = async () => {
    try {
      setUpdating(true);
      await axios.put(`http://127.0.0.1:5000/api/update/${selectedData?.id}`, {
        prompt: fields.prompt,
        completion: fields.completion,
      });
      await fetchData();
      alert('Data updated successfully');
      setUpdating(false);
    } catch (error) {
      console.error('Error updating data:', error);
    }
  };

  const handleSearch = (event: FormEvent) => {
    event.preventDefault();
    fetchData();
    setSelectedData(null);
  };

  const handleSelect = (data: Data) => {
    setSelectedData(data);
    setFields({
      id: data.id,
      prompt: data.prompt,
      completion: data.completion,
    });
  };

  const handleFieldChange = (event: ChangeEvent<HTMLInputElement | HTMLTextAreaElement>, key: keyof Data) => {
    setFields({ ...fields, [key]: event.target.value });
  };

  const handleUpdate = (event: FormEvent) => {
    event.preventDefault();
    updateData();
  };

  return (
    <div className="App">
      <div className="search-container" ref={searchContainerRef}>
        <h1>Text Data Labeller</h1>
        <form onSubmit={handleSearch} className="search-form">
          <input
            type="text"
            className="search-input"
            placeholder="Search dataset"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
          />
          <button type="submit" className="search-button">Search</button>
        </form>
      </div>
      <div className="content">
        {results.length > 0 && (
          <div className="results">
            <table>
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Prompt</th>
                  <th>Completion</th>
                </tr>
              </thead>
              <tbody>
                {results.map((result, index) => (
                  <tr key={index} onClick={() => handleSelect(result)}>
                    <td>{result.id}</td>
                    <td>{result.prompt}</td>
                    <td>{result.completion}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        {selectedData && (
            <div className="editor">
            <div>
              <label>id</label>
              <input
                type="text"
                value={fields.id}
                readOnly
              />
            </div>
            <div>
              <label>prompt</label>
              <textarea
                value={fields.prompt}
                onChange={(event) => handleFieldChange(event, 'prompt')}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && e.shiftKey) {
                    e.preventDefault();
                    const newVal = fields.prompt + '\n';
                    setFields({ ...fields, prompt: newVal });
                  }
                }}
              />
            </div>
            <div>
              <label>completion</label>
              <textarea
                value={fields.completion}
                onChange={(event) => handleFieldChange(event, 'completion')}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && e.shiftKey) {
                    e.preventDefault();
                    const newVal = fields.completion + '\n';
                    setFields({ ...fields, completion: newVal });
                  }
                }}
              />
            </div>
            <button onClick={handleUpdate} className="update-btn">{updating ? "updating.." : "update"}</button>
          </div>
        )}
      </div>
    </div>
  );
};

export default App;
        
