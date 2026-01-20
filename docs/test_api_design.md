# REST API Design Guidelines

## Overview

This document outlines our REST API design patterns, HTTP method usage, error handling, and versioning strategies.

## HTTP Methods

### GET - Retrieve Resources

- `GET /api/users` - List all users (paginated)
- `GET /api/users/:id` - Get single user by ID
- `GET /api/users/:id/posts` - Get user's posts

### POST - Create Resources

- `POST /api/users` - Create new user
- `POST /api/posts` - Create new post
- Request body contains resource data as JSON

### PUT - Full Update

- `PUT /api/users/:id` - Replace entire user record
- All fields required in request body

### PATCH - Partial Update

- `PATCH /api/users/:id` - Update specific fields
- Only changed fields in request body

### DELETE - Remove Resources

- `DELETE /api/users/:id` - Delete user
- Returns 204 No Content on success

## Response Formats

### Success Response

```json
{
  "status": "success",
  "data": { ... },
  "meta": {
    "page": 1,
    "total": 100
  }
}
```

### Error Response

```json
{
  "status": "error",
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Email is required",
    "details": [...]
  }
}
```

## Status Codes

- 200 OK - Successful GET, PUT, PATCH
- 201 Created - Successful POST
- 204 No Content - Successful DELETE
- 400 Bad Request - Validation error
- 401 Unauthorized - Missing or invalid auth
- 403 Forbidden - Insufficient permissions
- 404 Not Found - Resource doesn't exist
- 429 Too Many Requests - Rate limited
- 500 Internal Server Error - Server failure

## Pagination

Use cursor-based pagination for large datasets:

```
GET /api/posts?cursor=abc123&limit=20
```

Response includes:
- `next_cursor` for next page
- `has_more` boolean flag

## Versioning

API versioning via URL path:

- `/api/v1/users` - Version 1 (deprecated)
- `/api/v2/users` - Version 2 (current)
- `/api/v3/users` - Version 3 (beta)

## Rate Limiting

- 100 requests per minute for authenticated users
- 20 requests per minute for anonymous users
- X-RateLimit-Remaining header in responses
