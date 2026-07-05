//! Request-ID middleware. Injects `X-Request-ID` (UUID v4) on every request,
//! echoes it back in the response header, and adds it as a tracing span field
//! so logs can be correlated across the proxy boundary.

use axum::{
    body::Body,
    extract::Request,
    http::{HeaderName, HeaderValue},
    middleware::Next,
    response::Response,
};
use std::str::FromStr;

const REQUEST_ID_HEADER: &str = "x-request-id";

/// Axum middleware: stamp every request with an `X-Request-ID` (caller-supplied
/// or freshly generated UUID v4), expose it via a tracing span so all logs in
/// the request handler carry it, and echo it back on the response. Pattern that
/// lets you grep one request across the daemon, sglang, and llama.cpp logs.
pub async fn request_id_middleware(mut req: Request<Body>, next: Next) -> Response {
    // Use the caller's request ID if they supplied one (useful for tracing
    // across multiple services), else generate a fresh UUID.
    let header_name = HeaderName::from_str(REQUEST_ID_HEADER).expect("valid header name");
    let request_id = match req.headers().get(&header_name) {
        Some(v) if !v.is_empty() => v.to_str().unwrap_or("").to_string(),
        _ => uuid::Uuid::new_v4().to_string(),
    };

    // Inject into request extensions so handlers can read it.
    req.extensions_mut().insert(RequestId(request_id.clone()));

    // Open a tracing span for the duration of this request so all downstream
    // log lines get the request_id field automatically.
    let span = tracing::info_span!("http_request", request_id = %request_id);
    let _enter = span.enter();

    let mut response = next.run(req).await;

    // Echo back to client
    if let Ok(value) = HeaderValue::from_str(&request_id) {
        response.headers_mut().insert(header_name, value);
    }
    response
}

/// Newtype wrapper so handlers can pull the request ID out of
/// `Request::extensions` with `req.extensions().get::<RequestId>()`. The
/// wrapper avoids the awkwardness of fishing for a raw `String`.
#[derive(Debug, Clone)]
pub struct RequestId(pub String);
