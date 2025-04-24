import React, { useEffect, useState } from 'react';
import { w3cwebsocket as W3CWebSocket } from 'websocket';

const WebSocketHandler = ({ backtestId, onUpdate }) => {
  const [client, setClient] = useState(null);

  useEffect(() => {
    const newClient = new W3CWebSocket(`ws://localhost:8000/ws/backtest/${backtestId}`);

    newClient.onopen = () => {
      console.log('WebSocket Client Connected');
    };

    newClient.onmessage = (message) => {
      const data = JSON.parse(message.data);
      onUpdate(data);
    };

    newClient.onclose = () => {
      console.log('WebSocket Client Disconnected');
    };

    setClient(newClient);

    return () => {
      newClient.close();
    };
  }, [backtestId, onUpdate]);

  return null; // This component doesn't render anything
};

export default WebSocketHandler;