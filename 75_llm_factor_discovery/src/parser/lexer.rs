//! Factor expression lexer
//!
//! Tokenizes factor expressions for parsing.

use thiserror::Error;

/// Token kinds in factor expressions
#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    /// Identifier (function name or variable)
    Ident(String),
    /// Numeric literal
    Number(f64),
    /// Opening parenthesis
    LParen,
    /// Closing parenthesis
    RParen,
    /// Comma separator
    Comma,
    /// Plus operator
    Plus,
    /// Minus operator
    Minus,
    /// Multiply operator
    Star,
    /// Divide operator
    Slash,
    /// End of input
    Eof,
}

/// A token with position information
#[derive(Debug, Clone)]
pub struct Token {
    pub kind: TokenKind,
    pub position: usize,
    pub length: usize,
}

impl Token {
    pub fn new(kind: TokenKind, position: usize, length: usize) -> Self {
        Self {
            kind,
            position,
            length,
        }
    }
}

/// Lexer error types
#[derive(Error, Debug, Clone)]
pub enum LexerError {
    #[error("Unexpected character '{0}' at position {1}")]
    UnexpectedChar(char, usize),

    #[error("Invalid number at position {0}")]
    InvalidNumber(usize),
}

/// Lexer for factor expressions
pub struct FactorLexer<'a> {
    input: &'a str,
    position: usize,
    chars: std::iter::Peekable<std::str::CharIndices<'a>>,
}

impl<'a> FactorLexer<'a> {
    /// Create a new lexer for the given input
    pub fn new(input: &'a str) -> Self {
        Self {
            input,
            position: 0,
            chars: input.char_indices().peekable(),
        }
    }

    /// Tokenize the entire input
    pub fn tokenize(&mut self) -> Result<Vec<Token>, LexerError> {
        let mut tokens = Vec::new();

        loop {
            let token = self.next_token()?;
            let is_eof = token.kind == TokenKind::Eof;
            tokens.push(token);
            if is_eof {
                break;
            }
        }

        Ok(tokens)
    }

    /// Get the next token
    pub fn next_token(&mut self) -> Result<Token, LexerError> {
        self.skip_whitespace();

        let Some(&(pos, ch)) = self.chars.peek() else {
            return Ok(Token::new(TokenKind::Eof, self.input.len(), 0));
        };

        match ch {
            '(' => {
                self.chars.next();
                Ok(Token::new(TokenKind::LParen, pos, 1))
            }
            ')' => {
                self.chars.next();
                Ok(Token::new(TokenKind::RParen, pos, 1))
            }
            ',' => {
                self.chars.next();
                Ok(Token::new(TokenKind::Comma, pos, 1))
            }
            '+' => {
                self.chars.next();
                Ok(Token::new(TokenKind::Plus, pos, 1))
            }
            '-' => {
                // Could be negative number or minus operator
                self.chars.next();
                if let Some(&(_, next_ch)) = self.chars.peek() {
                    if next_ch.is_ascii_digit() || next_ch == '.' {
                        // It's a negative number
                        let (num, len) = self.read_number(pos)?;
                        return Ok(Token::new(TokenKind::Number(-num), pos, len + 1));
                    }
                }
                Ok(Token::new(TokenKind::Minus, pos, 1))
            }
            '*' => {
                self.chars.next();
                Ok(Token::new(TokenKind::Star, pos, 1))
            }
            '/' => {
                self.chars.next();
                Ok(Token::new(TokenKind::Slash, pos, 1))
            }
            '0'..='9' | '.' => {
                let (num, len) = self.read_number(pos)?;
                Ok(Token::new(TokenKind::Number(num), pos, len))
            }
            'a'..='z' | 'A'..='Z' | '_' => {
                let (ident, len) = self.read_identifier();
                Ok(Token::new(TokenKind::Ident(ident), pos, len))
            }
            _ => Err(LexerError::UnexpectedChar(ch, pos)),
        }
    }

    fn skip_whitespace(&mut self) {
        while let Some(&(_, ch)) = self.chars.peek() {
            if ch.is_whitespace() {
                self.chars.next();
            } else {
                break;
            }
        }
    }

    fn read_number(&mut self, start_pos: usize) -> Result<(f64, usize), LexerError> {
        let mut num_str = String::new();
        let mut has_dot = false;

        while let Some(&(_, ch)) = self.chars.peek() {
            match ch {
                '0'..='9' => {
                    num_str.push(ch);
                    self.chars.next();
                }
                '.' if !has_dot => {
                    has_dot = true;
                    num_str.push(ch);
                    self.chars.next();
                }
                'e' | 'E' => {
                    // Scientific notation
                    num_str.push(ch);
                    self.chars.next();
                    // Handle optional sign
                    if let Some(&(_, sign)) = self.chars.peek() {
                        if sign == '+' || sign == '-' {
                            num_str.push(sign);
                            self.chars.next();
                        }
                    }
                }
                _ => break,
            }
        }

        let len = num_str.len();
        num_str
            .parse::<f64>()
            .map(|n| (n, len))
            .map_err(|_| LexerError::InvalidNumber(start_pos))
    }

    fn read_identifier(&mut self) -> (String, usize) {
        let mut ident = String::new();

        while let Some(&(_, ch)) = self.chars.peek() {
            if ch.is_alphanumeric() || ch == '_' {
                ident.push(ch);
                self.chars.next();
            } else {
                break;
            }
        }

        let len = ident.len();
        (ident, len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize_simple() {
        let mut lexer = FactorLexer::new("rank(close)");
        let tokens = lexer.tokenize().unwrap();

        assert_eq!(tokens.len(), 5); // rank, (, close, ), EOF
        assert!(matches!(tokens[0].kind, TokenKind::Ident(ref s) if s == "rank"));
        assert_eq!(tokens[1].kind, TokenKind::LParen);
        assert!(matches!(tokens[2].kind, TokenKind::Ident(ref s) if s == "close"));
        assert_eq!(tokens[3].kind, TokenKind::RParen);
        assert_eq!(tokens[4].kind, TokenKind::Eof);
    }

    #[test]
    fn test_tokenize_numbers() {
        let mut lexer = FactorLexer::new("ts_mean(close, 20)");
        let tokens = lexer.tokenize().unwrap();

        assert!(matches!(tokens[3].kind, TokenKind::Number(n) if (n - 20.0).abs() < 1e-10));
    }

    #[test]
    fn test_tokenize_negative_number() {
        let mut lexer = FactorLexer::new("-5");
        let tokens = lexer.tokenize().unwrap();

        assert!(matches!(tokens[0].kind, TokenKind::Number(n) if (n - (-5.0)).abs() < 1e-10));
    }

    #[test]
    fn test_tokenize_scientific_notation() {
        let mut lexer = FactorLexer::new("1.5e-3");
        let tokens = lexer.tokenize().unwrap();

        assert!(matches!(tokens[0].kind, TokenKind::Number(n) if (n - 0.0015).abs() < 1e-10));
    }
}
