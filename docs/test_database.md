# Database Schema Design

## Overview

This document describes our PostgreSQL database design patterns, indexing strategies, and query optimization techniques.

## Schema Design

### Users Table

```sql
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(100),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    deleted_at TIMESTAMPTZ
);
```

### Posts Table

```sql
CREATE TABLE posts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    author_id UUID REFERENCES users(id),
    title VARCHAR(200) NOT NULL,
    content TEXT,
    status VARCHAR(20) DEFAULT 'draft',
    published_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

### Tags and Post-Tags (Many-to-Many)

```sql
CREATE TABLE tags (
    id SERIAL PRIMARY KEY,
    name VARCHAR(50) UNIQUE NOT NULL
);

CREATE TABLE post_tags (
    post_id UUID REFERENCES posts(id),
    tag_id INT REFERENCES tags(id),
    PRIMARY KEY (post_id, tag_id)
);
```

## Indexing Strategies

### B-Tree Indexes

Default index type for equality and range queries:

```sql
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_posts_author ON posts(author_id);
CREATE INDEX idx_posts_status_published ON posts(status, published_at);
```

### GIN Indexes for Full-Text Search

```sql
CREATE INDEX idx_posts_content_fts ON posts
USING GIN (to_tsvector('english', content));
```

### Partial Indexes

Index only relevant rows:

```sql
CREATE INDEX idx_posts_published ON posts(published_at)
WHERE status = 'published';
```

## Query Optimization

### EXPLAIN ANALYZE

Always analyze slow queries:

```sql
EXPLAIN ANALYZE SELECT * FROM posts
WHERE author_id = '123' ORDER BY created_at DESC;
```

### Common Table Expressions (CTEs)

Use CTEs for complex queries:

```sql
WITH recent_posts AS (
    SELECT * FROM posts
    WHERE created_at > NOW() - INTERVAL '7 days'
)
SELECT u.full_name, COUNT(p.id)
FROM users u
JOIN recent_posts p ON p.author_id = u.id
GROUP BY u.id;
```

### Connection Pooling

- Use PgBouncer for connection pooling
- Transaction pooling mode for web apps
- Pool size = (cores * 2) + effective_spindle_count

## Migrations

Use numbered migration files:

- `001_create_users.sql`
- `002_create_posts.sql`
- `003_add_user_roles.sql`

## Backup Strategy

- Daily full backups with pg_dump
- Continuous WAL archiving to S3
- Point-in-time recovery capability
- Test restores monthly
