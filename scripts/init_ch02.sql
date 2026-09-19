SET NAMES utf8mb4;

CREATE TABLE IF NOT EXISTS faq (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    question VARCHAR(255) NOT NULL,
    answer TEXT NOT NULL,
    category VARCHAR(64) NOT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    UNIQUE KEY uq_faq_question (question),
    KEY idx_faq_category (category)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS conversations (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    session_id VARCHAR(34) NOT NULL,
    user_id VARCHAR(64) NULL,
    status VARCHAR(32) NOT NULL DEFAULT 'open',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    UNIQUE KEY uq_conversations_session_id (session_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS messages (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    conversation_id BIGINT UNSIGNED NOT NULL,
    role VARCHAR(16) NOT NULL,
    content TEXT NULL,
    tool_name VARCHAR(64) NULL,
    tool_call_id VARCHAR(128) NULL,
    tool_arguments JSON NULL,
    tool_result JSON NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    KEY idx_messages_conversation_created (conversation_id, created_at),
    KEY idx_messages_tool_call (tool_call_id),
    CONSTRAINT fk_messages_conversation
        FOREIGN KEY (conversation_id) REFERENCES conversations (id)
        ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS tickets (
    ticket_id VARCHAR(32) NOT NULL,
    conversation_id BIGINT UNSIGNED NOT NULL,
    description TEXT NOT NULL,
    ticket_type VARCHAR(32) NOT NULL,
    status VARCHAR(32) NOT NULL DEFAULT 'pending',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ticket_id),
    KEY idx_tickets_conversation_created (conversation_id, created_at),
    KEY idx_tickets_status (status),
    CONSTRAINT fk_tickets_conversation
        FOREIGN KEY (conversation_id) REFERENCES conversations (id)
        ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

INSERT INTO faq (question, answer, category) VALUES
    ('退货政策是什么', '签收后 7 天内可申请退货，商品需保持完好并符合平台规则。', 'policy'),
    ('如何申请换货', '请在订单详情中提交换货申请，并提供订单号、商品问题和期望方案。', 'after_sales'),
    ('发票怎么开', '可在订单详情中申请电子发票，具体开票时间以平台规则为准。', 'invoice'),
    ('物流多久到货', '常规地区通常 2 到 5 天送达，偏远地区时间可能更长。', 'logistics'),
    ('退款多久到账', '退款审核通过后通常 1 到 7 个工作日到账，具体以支付渠道为准。', 'refund')
ON DUPLICATE KEY UPDATE
    answer = VALUES(answer),
    category = VALUES(category);
