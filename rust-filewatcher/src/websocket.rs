/*!
 * WebSocket server for real-time dashboard updates
 *
 * Provides real-time streaming of file changes, analysis results, and system status
 * to development dashboards and other clients using WebSocket connections.
 */

use crate::types::*;

use anyhow::Result;
use crossbeam::channel::{Receiver, Sender};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{broadcast, RwLock};
use tokio_tungstenite::{
    accept_async,
    tungstenite::{protocol::Message, Error as WsError},
    WebSocketStream,
};
use tracing::{debug, error, info, trace, warn};
use uuid::Uuid;

/// WebSocket server for real-time updates
pub struct WebSocketServer {
    /// Server address
    addr: SocketAddr,

    /// Broadcast channel for sending updates to all clients
    broadcast_tx: broadcast::Sender<DashboardUpdate>,

    /// Connected clients
    clients: Arc<RwLock<HashMap<Uuid, ClientConnection>>>,

    /// Connection counter
    connection_counter: Arc<AtomicUsize>,
}

/// Information about a connected client
#[derive(Debug, Clone)]
pub struct ClientConnection {
    /// Unique client ID
    pub id: Uuid,

    /// Client address
    pub addr: SocketAddr,

    /// When the client connected
    pub connected_at: chrono::DateTime<chrono::Utc>,

    /// Client subscription preferences
    pub subscriptions: ClientSubscriptions,

    /// Channel for sending messages to this client
    pub sender: tokio::sync::mpsc::UnboundedSender<Message>,
}

/// Client subscription preferences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientSubscriptions {
    /// Subscribe to file change events
    #[serde(default = "default_true")]
    pub file_changes: bool,

    /// Subscribe to analysis results
    #[serde(default = "default_true")]
    pub analysis_results: bool,

    /// Subscribe to system status updates
    #[serde(default = "default_true")]
    pub system_status: bool,

    /// Subscribe to performance metrics
    #[serde(default)]
    pub performance_metrics: bool,

    /// Subscribe to error notifications
    #[serde(default = "default_true")]
    pub error_notifications: bool,

    /// File path filters (only send updates for matching paths)
    #[serde(default)]
    pub path_filters: Vec<String>,

    /// Tool filters (only send updates for specific tools)
    #[serde(default)]
    pub tool_filters: Vec<String>,
}

/// WebSocket message types for client communication
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum WebSocketMessage {
    /// Client subscription update
    Subscribe {
        subscriptions: ClientSubscriptions,
    },

    /// Ping/pong for connection health
    Ping {
        timestamp: chrono::DateTime<chrono::Utc>,
    },

    Pong {
        timestamp: chrono::DateTime<chrono::Utc>,
    },

    /// Dashboard update from server
    Update {
        update: DashboardUpdate,
    },

    /// Error message
    Error {
        message: String,
    },

    /// Connection acknowledgment
    Connected {
        client_id: Uuid,
        server_version: String,
    },

    /// Heartbeat for connection maintenance
    Heartbeat,
}

fn default_true() -> bool {
    true
}

impl Default for ClientSubscriptions {
    fn default() -> Self {
        Self {
            file_changes: true,
            analysis_results: true,
            system_status: true,
            performance_metrics: false,
            error_notifications: true,
            path_filters: Vec::new(),
            tool_filters: Vec::new(),
        }
    }
}

impl WebSocketServer {
    /// Create a new WebSocket server
    pub fn new(addr: SocketAddr) -> Self {
        let (broadcast_tx, _) = broadcast::channel(1000);

        Self {
            addr,
            broadcast_tx,
            clients: Arc::new(RwLock::new(HashMap::new())),
            connection_counter: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Get a sender for broadcasting updates
    pub fn get_update_sender(&self) -> Sender<DashboardUpdate> {
        let broadcast_tx = self.broadcast_tx.clone();
        let (tx, rx) = crossbeam::channel::unbounded();

        // Spawn task to forward messages from crossbeam to broadcast channel
        tokio::spawn(async move {
            while let Ok(update) = rx.recv() {
                if let Err(e) = broadcast_tx.send(update) {
                    warn!("Failed to broadcast update: {}", e);
                }
            }
        });

        tx
    }

    /// Start the WebSocket server
    pub async fn start(self: Arc<Self>) -> Result<()> {
        info!("🌐 Starting WebSocket server on {}", self.addr);

        let listener = TcpListener::bind(self.addr).await?;

        // Spawn heartbeat task
        let server_clone = self.clone();
        tokio::spawn(async move {
            server_clone.heartbeat_task().await;
        });

        // Spawn cleanup task
        let server_clone = self.clone();
        tokio::spawn(async move {
            server_clone.cleanup_task().await;
        });

        info!("✅ WebSocket server listening on {}", self.addr);

        while let Ok((stream, addr)) = listener.accept().await {
            let server_clone = self.clone();
            tokio::spawn(async move {
                if let Err(e) = server_clone.handle_connection(stream, addr).await {
                    warn!("WebSocket connection error from {}: {}", addr, e);
                }
            });
        }

        Ok(())
    }

    /// Handle a new WebSocket connection
    async fn handle_connection(&self, stream: TcpStream, addr: SocketAddr) -> Result<()> {
        let connection_id = self.connection_counter.fetch_add(1, Ordering::Relaxed);
        info!("🔗 New WebSocket connection #{} from {}", connection_id, addr);

        // Upgrade to WebSocket
        let ws_stream = match accept_async(stream).await {
            Ok(ws) => ws,
            Err(e) => {
                warn!("Failed to upgrade WebSocket connection: {}", e);
                return Err(e.into());
            }
        };

        let client_id = Uuid::new_v4();

        // Create channels for this client
        let (client_tx, mut client_rx) = tokio::sync::mpsc::unbounded_channel();

        // Create client connection info
        let client = ClientConnection {
            id: client_id,
            addr,
            connected_at: chrono::Utc::now(),
            subscriptions: ClientSubscriptions::default(),
            sender: client_tx,
        };

        // Add client to connected clients
        {
            let mut clients = self.clients.write().await;
            clients.insert(client_id, client);
        }

        // Send connection acknowledgment
        let connected_msg = WebSocketMessage::Connected {
            client_id,
            server_version: env!("CARGO_PKG_VERSION").to_string(),
        };

        if let Ok(msg_text) = serde_json::to_string(&connected_msg) {
            let _ = client_rx.recv(); // Consume the first message slot
            if let Err(e) = client_rx.try_recv() {
                debug!("Client channel setup: {:?}", e);
            }
        }

        // Split the WebSocket stream
        let (mut ws_sender, mut ws_receiver) = ws_stream.split();

        // Subscribe to broadcast channel
        let mut broadcast_rx = self.broadcast_tx.subscribe();

        // Spawn task to send messages to client
        let clients_clone = self.clients.clone();
        let client_id_clone = client_id;
        let send_task = tokio::spawn(async move {
            loop {
                tokio::select! {
                    // Receive from client-specific channel
                    msg = client_rx.recv() => {
                        if let Some(msg) = msg {
                            use futures_util::SinkExt;
                        if let Err(e) = ws_sender.send(msg).await {
                                warn!("Failed to send message to client {}: {}", client_id_clone, e);
                                break;
                            }
                        } else {
                            debug!("Client {} channel closed", client_id_clone);
                            break;
                        }
                    }

                    // Receive from broadcast channel
                    update = broadcast_rx.recv() => {
                        if let Ok(update) = update {
                            // Check if client should receive this update
                            let should_send = {
                                let clients = clients_clone.read().await;
                                if let Some(client) = clients.get(&client_id_clone) {
                                    Self::should_send_update(&update, &client.subscriptions)
                                } else {
                                    false
                                }
                            };

                            if should_send {
                                let ws_msg = WebSocketMessage::Update { update };
                                if let Ok(msg_text) = serde_json::to_string(&ws_msg) {
                                    let text_msg = Message::Text(msg_text);
                                    if let Err(e) = ws_sender.send(text_msg).await {
                                        warn!("Failed to send update to client {}: {}", client_id_clone, e);
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        });

        // Handle incoming messages from client
        use futures_util::StreamExt;
        while let Some(msg_result) = ws_receiver.next().await {
            match msg_result {
                Ok(msg) => {
                    if let Err(e) = self.handle_client_message(client_id, msg).await {
                        warn!("Error handling message from client {}: {}", client_id, e);
                    }
                }
                Err(WsError::ConnectionClosed) => {
                    info!("Client {} disconnected normally", client_id);
                    break;
                }
                Err(e) => {
                    warn!("WebSocket error from client {}: {}", client_id, e);
                    break;
                }
            }
        }

        // Cleanup
        send_task.abort();
        {
            let mut clients = self.clients.write().await;
            clients.remove(&client_id);
        }

        info!("🔌 Client {} disconnected", client_id);
        Ok(())
    }

    /// Handle a message from a client
    async fn handle_client_message(&self, client_id: Uuid, message: Message) -> Result<()> {
        match message {
            Message::Text(text) => {
                trace!("Received text message from {}: {}", client_id, text);

                // Parse as WebSocket message
                match serde_json::from_str::<WebSocketMessage>(&text) {
                    Ok(ws_msg) => {
                        self.process_client_message(client_id, ws_msg).await?;
                    }
                    Err(e) => {
                        warn!("Invalid JSON message from client {}: {}", client_id, e);
                        self.send_error_to_client(client_id, "Invalid message format").await?;
                    }
                }
            }

            Message::Ping(payload) => {
                trace!("Received ping from client {}", client_id);
                self.send_pong_to_client(client_id, payload).await?;
            }

            Message::Pong(_) => {
                trace!("Received pong from client {}", client_id);
            }

            Message::Close(_) => {
                info!("Client {} sent close message", client_id);
            }

            Message::Binary(_) => {
                warn!("Binary messages not supported from client {}", client_id);
                self.send_error_to_client(client_id, "Binary messages not supported").await?;
            }

            Message::Frame(_) => {
                // Internal message type, ignore
            }
        }

        Ok(())
    }

    /// Process a parsed WebSocket message from client
    async fn process_client_message(&self, client_id: Uuid, message: WebSocketMessage) -> Result<()> {
        match message {
            WebSocketMessage::Subscribe { subscriptions } => {
                info!("Client {} updated subscriptions", client_id);

                let mut clients = self.clients.write().await;
                if let Some(client) = clients.get_mut(&client_id) {
                    client.subscriptions = subscriptions;
                }
            }

            WebSocketMessage::Ping { timestamp } => {
                let pong = WebSocketMessage::Pong { timestamp };
                self.send_message_to_client(client_id, pong).await?;
            }

            WebSocketMessage::Heartbeat => {
                // Client heartbeat, no action needed
                trace!("Heartbeat from client {}", client_id);
            }

            _ => {
                warn!("Unexpected message type from client {}", client_id);
            }
        }

        Ok(())
    }

    /// Send a message to a specific client
    async fn send_message_to_client(&self, client_id: Uuid, message: WebSocketMessage) -> Result<()> {
        let clients = self.clients.read().await;
        if let Some(client) = clients.get(&client_id) {
            let msg_text = serde_json::to_string(&message)?;
            let ws_message = Message::Text(msg_text);

            if let Err(e) = client.sender.send(ws_message) {
                warn!("Failed to queue message for client {}: {}", client_id, e);
            }
        }

        Ok(())
    }

    /// Send an error message to a client
    async fn send_error_to_client(&self, client_id: Uuid, error: &str) -> Result<()> {
        let error_msg = WebSocketMessage::Error {
            message: error.to_string(),
        };

        self.send_message_to_client(client_id, error_msg).await
    }

    /// Send a pong response to a client
    async fn send_pong_to_client(&self, client_id: Uuid, payload: Vec<u8>) -> Result<()> {
        let clients = self.clients.read().await;
        if let Some(client) = clients.get(&client_id) {
            let pong_message = Message::Pong(payload);

            if let Err(e) = client.sender.send(pong_message) {
                warn!("Failed to send pong to client {}: {}", client_id, e);
            }
        }

        Ok(())
    }

    /// Check if an update should be sent to a client based on subscriptions
    fn should_send_update(update: &DashboardUpdate, subscriptions: &ClientSubscriptions) -> bool {
        match &update.update_type {
            DashboardUpdateType::FileChanged => subscriptions.file_changes,
            DashboardUpdateType::AnalysisCompleted => subscriptions.analysis_results,
            DashboardUpdateType::AnalysisStarted => subscriptions.analysis_results,
            DashboardUpdateType::StatusUpdate => subscriptions.system_status,
            DashboardUpdateType::Error => subscriptions.error_notifications,
            DashboardUpdateType::BatchUpdate => subscriptions.analysis_results,
            DashboardUpdateType::ToolUpdate => subscriptions.analysis_results,
        }
    }

    /// Periodic heartbeat task
    async fn heartbeat_task(&self) {
        let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(30));

        loop {
            interval.tick().await;

            let clients = self.clients.read().await;
            let client_count = clients.len();

            if client_count > 0 {
                debug!("Sending heartbeat to {} clients", client_count);

                for client_id in clients.keys() {
                    let heartbeat_msg = WebSocketMessage::Heartbeat;
                    if let Ok(msg_text) = serde_json::to_string(&heartbeat_msg) {
                        let ws_message = Message::Text(msg_text);
                        if let Some(client) = clients.get(client_id) {
                            let _ = client.sender.send(ws_message);
                        }
                    }
                }
            }
        }
    }

    /// Periodic cleanup task for stale connections
    async fn cleanup_task(&self) {
        let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(300)); // 5 minutes

        loop {
            interval.tick().await;

            let mut clients_to_remove = Vec::new();
            let cutoff = chrono::Utc::now() - chrono::Duration::minutes(30);

            {
                let clients = self.clients.read().await;
                for (client_id, client) in clients.iter() {
                    if client.connected_at < cutoff {
                        clients_to_remove.push(*client_id);
                    }
                }
            }

            if !clients_to_remove.is_empty() {
                let mut clients = self.clients.write().await;
                for client_id in clients_to_remove {
                    clients.remove(&client_id);
                    info!("Cleaned up stale client connection: {}", client_id);
                }
            }
        }
    }

    /// Get connected client count
    pub async fn get_client_count(&self) -> usize {
        self.clients.read().await.len()
    }

    /// Get client information
    pub async fn get_clients(&self) -> Vec<ClientConnection> {
        self.clients.read().await.values().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{IpAddr, Ipv4Addr};

    #[test]
    fn test_client_subscriptions_default() {
        let subs = ClientSubscriptions::default();
        assert!(subs.file_changes);
        assert!(subs.analysis_results);
        assert!(subs.system_status);
        assert!(!subs.performance_metrics);
        assert!(subs.error_notifications);
    }

    #[test]
    fn test_should_send_update() {
        let subs = ClientSubscriptions {
            file_changes: true,
            analysis_results: false,
            ..Default::default()
        };

        let file_update = DashboardUpdate {
            update_type: DashboardUpdateType::FileChanged,
            timestamp: chrono::Utc::now(),
            data: DashboardData::FileChange {
                file: std::path::PathBuf::from("test.py"),
                event_type: FileEventType::Modify,
            },
        };

        assert!(WebSocketServer::should_send_update(&file_update, &subs));

        let analysis_update = DashboardUpdate {
            update_type: DashboardUpdateType::AnalysisCompleted,
            timestamp: chrono::Utc::now(),
            data: DashboardData::Analysis(AnalysisResult::new(std::path::PathBuf::from("test.py"))),
        };

        assert!(!WebSocketServer::should_send_update(&analysis_update, &subs));
    }

    #[tokio::test]
    async fn test_websocket_server_creation() {
        let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 8768);
        let server = Arc::new(WebSocketServer::new(addr));

        assert_eq!(server.get_client_count().await, 0);
    }
}
