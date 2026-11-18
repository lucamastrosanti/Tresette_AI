# Tresette AI

Gioco di Tressette con interfaccia React e backend FastAPI. Il bot combina un modello
`MaskablePPO`, simulazioni PIMC e una fase finale minimax perfetta.

## Requisiti

- Python 3.10 o 3.11
- Node.js 20 o superiore
- npm

## Avvio

### 1. Backend

Da questa cartella:

```powershell
py -m pip install -r requirements.txt
py -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Il backend cerca il modello `tressette_bot_supremo_assoluto.zip` nella cartella radice.

Endpoint principali:

- API interattiva: `http://127.0.0.1:8000/docs`
- nuova partita: `POST http://127.0.0.1:8000/api/nuova_partita`

### 2. Frontend

In un secondo terminale:

```powershell
cd frontend
npm install
npm run dev
```

Apri l'indirizzo mostrato da Vite, normalmente `http://localhost:5173`.
Il frontend usa `VITE_API_URL` se configurata, altrimenti si collega a
`http://127.0.0.1:8000`.

Per configurare un backend diverso, crea `frontend/.env.local`:

```text
VITE_API_URL=http://127.0.0.1:8000
```

## Struttura

- `api.py`: server FastAPI, gestione sessioni e bot ibrido
- `env.py`: ambiente Gymnasium e bot euristico
- `engine.py`: carte, mazzo, prese e regole del Tressette
- `train.py`: addestramento/evoluzione del modello
- `tressette_bot_supremo_assoluto.zip`: modello pre-addestrato richiesto dal backend
- `frontend/`: interfaccia React/Vite
- `requirements.txt`: dipendenze Python
- `log_partite.json` e `storico_test.txt`: statistiche generate dal gioco

## Note

- Il frontend richiede il backend FastAPI; non e' un progetto Streamlit.
- Le statistiche vengono salvate in locale e, se disponibile, anche su Supabase.
- `node_modules/` e `__pycache__/` sono directory generate e non devono essere
  aggiunte al repository.