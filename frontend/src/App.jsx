import { useState } from 'react'

function App() {
  const apiUrl = (import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000').replace(/\/$/, '')
  const [gameState, setGameState] = useState({
    inGame: false,
    mano: [],
    carte_sul_tavolo: [], 
    punti_umano: 0,
    punti_ai: 0,
    blocco_click: false, 
    messaggio: "Premi 'Inizia Partita' per sfidare l'IA!",
    session_id: null
  });

  const startNuovaPartita = async () => {
    try {
      setGameState(prev => ({...prev, messaggio: "Connessione al tavolo...", blocco_click: true}));
      const response = await fetch(`${apiUrl}/api/nuova_partita`, { method: 'POST' });
      const data = await response.json();
      
      setGameState({
        inGame: true,
        mano: data.mano,
        carte_sul_tavolo: data.carta_in_tavola ? [data.carta_in_tavola] : [],
        punti_umano: 0,
        punti_ai: 0,
        blocco_click: false, 
        messaggio: "Partita iniziata! Tocca a te.",
        session_id: data.session_id
      });
    } catch (error) {
      console.error("Errore:", error);
      setGameState(prev => ({...prev, messaggio: "❌ Errore di connessione.", blocco_click: false}));
    }
  };

  const giocaCarta = async (id_carta) => {
    if (gameState.blocco_click) return;

    try {
      const cartaGiocata = gameState.mano.find(c => c.id === id_carta);
      const tavoloAggiornato = [...gameState.carte_sul_tavolo, cartaGiocata];

      setGameState(prev => ({
        ...prev,
        blocco_click: true, 
        messaggio: "L'IA sta riflettendo...",
        mano: prev.mano.filter(c => c.id !== id_carta),
        carte_sul_tavolo: tavoloAggiornato
      }));
      
      const response = await fetch(`${apiUrl}/api/gioca_carta`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
            id_carta: id_carta,
            session_id: gameState.session_id
        })
      });
      
      const data = await response.json();
      
      if (data.errore) {
        setGameState(prev => ({
          ...prev,
          messaggio: "❌ " + data.messaggio,
          blocco_click: false,
          mano: [...prev.mano, cartaGiocata].sort((a, b) => a.id - b.id),
          carte_sul_tavolo: prev.carte_sul_tavolo.filter(c => c.id !== id_carta) 
        }));
        return; 
      }
      
      if (response.ok) {
        const tavoloFinale = data.carta_giocata_da_ai 
          ? [...tavoloAggiornato, data.carta_giocata_da_ai] 
          : tavoloAggiornato;

        setGameState(prev => ({
          ...prev,
          carte_sul_tavolo: tavoloFinale, 
          messaggio: "L'IA ha giocato! Risoluzione..."
        }));

        setTimeout(() => {
          // CORREZIONE NOMI: Se il backend restituisce "Agente_AI", significa che ha preso il giocatore
          let msgTurno = "Tocca a te!";
          if (data.ultimo_vincitore) {
            const nomeVincitore = data.ultimo_vincitore === "Agente_AI" ? "Te" : "L'IA";
            msgTurno = `Presa da: ${nomeVincitore}. Tocca a te!`;
          }

          setGameState(prev => ({
            ...prev,
            inGame: !data.finito,
            mano: data.mano,
            carte_sul_tavolo: data.nuova_carta_in_tavola ? [data.nuova_carta_in_tavola] : [],
            punti_umano: data.punti_umano,
            punti_ai: data.punti_ai,
            blocco_click: false,
            messaggio: data.finito 
              ? `Fine! Tu: ${Number(data.punti_umano).toFixed(2)} - IA: ${Number(data.punti_ai).toFixed(2)}`
              : msgTurno
          }));
        }, 2000);

      } else {
        setGameState(prev => ({...prev, messaggio: "Errore dal server!", blocco_click: false}));
      }
    } catch (error) {
      console.error("Errore:", error);
      setGameState(prev => ({...prev, messaggio: "❌ Errore di connessione.", blocco_click: false}));
    }
  };

  return (
    <div className="min-h-screen bg-green-800 flex flex-col items-center justify-between p-2 sm:p-8 text-white font-sans overflow-hidden">
      
      {/* HEADER PUNTEGGI (Responsive) */}
      <div className="w-full max-w-5xl flex justify-between items-center bg-green-900 p-2 sm:p-4 rounded-xl shadow-xl border-2 border-green-700">
        <div className="text-sm sm:text-2xl font-bold bg-black/20 px-2 sm:px-4 py-1 sm:py-2 rounded-lg whitespace-nowrap">Tu: {Number(gameState.punti_umano).toFixed(2)}</div>
        <div className="text-xs sm:text-xl font-medium text-yellow-300 tracking-wide text-center px-1 sm:px-2 leading-tight w-full max-w-[160px] sm:max-w-md">{gameState.messaggio}</div>
        <div className="text-sm sm:text-2xl font-bold bg-black/20 px-2 sm:px-4 py-1 sm:py-2 rounded-lg whitespace-nowrap">IA: {Number(gameState.punti_ai).toFixed(2)}</div>
      </div>

      {/* TAVOLO (Responsive) */}
      <div className="flex-grow flex items-center justify-center gap-2 sm:gap-6 w-full max-w-5xl my-4 sm:my-8 border-2 sm:border-4 border-dashed border-green-600/50 rounded-2xl sm:rounded-3xl bg-green-900/30 relative min-h-[160px] sm:min-h-[250px]">
        {gameState.carte_sul_tavolo.length > 0 ? (
          gameState.carte_sul_tavolo.map((carta, index) => (
            <img 
              key={index}
              src={carta?.id !== undefined ? `/carte/${carta.id}.png` : '/carte/card_back.png'} 
              alt={carta?.nome || "Carta in tavola"}
              className="w-20 sm:w-32 h-auto object-contain rounded-lg sm:rounded-xl shadow-2xl drop-shadow-[0_10px_10px_rgba(0,0,0,0.5)] border border-black/20 transform scale-105"
              onError={(e) => e.target.src='/carte/card_back.png'}
            />
          ))
        ) : (
          <div className="text-green-500/50 text-xl sm:text-3xl font-semibold uppercase tracking-widest">Tavolo</div>
        )}
      </div>

      {/* MANO (Compatta e non wrap per Mobile) */}
      <div className="w-full max-w-5xl flex flex-col items-center justify-end h-32 sm:h-56 pb-2 sm:pb-4">
        {!gameState.inGame ? (
          <button 
            onClick={startNuovaPartita}
            className="px-6 sm:px-10 py-3 sm:py-5 bg-gradient-to-r from-yellow-600 to-yellow-500 hover:from-yellow-500 hover:to-yellow-400 text-white font-extrabold rounded-full text-lg sm:text-2xl shadow-2xl transition-all transform hover:scale-105 active:scale-95 uppercase tracking-wider whitespace-nowrap"
          >
            Inizia Partita
          </button>
        ) : (
          <div className="flex justify-center items-end -space-x-8 sm:-space-x-12 w-full px-1">
            {gameState.mano.map((carta, index) => (
              <img 
                key={index} 
                src={`/carte/${carta.id}.png`}
                alt={carta.nome}
                className={`w-16 sm:w-28 h-auto object-contain rounded-md sm:rounded-xl shadow-xl transition-all duration-200 border sm:border-2 relative flex-shrink-0 
                  ${gameState.blocco_click 
                    ? 'opacity-70 cursor-not-allowed border-transparent' 
                    : 'cursor-pointer hover:-translate-y-4 sm:hover:-translate-y-8 border-transparent hover:border-yellow-400 hover:z-50'}`}
                onClick={() => giocaCarta(carta.id)}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

export default App