import logging
import sys

def setup_logging(level=logging.INFO):
    """
    Configura o logging global da aplicação.
    Deve ser chamado apenas uma vez no ponto de entrada (entrypoint).
    """
    # Define o formato: Data | Nivel | Arquivo de Origem | Mensagem
    log_format = '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    formatter = logging.Formatter(fmt=log_format, datefmt=date_format)

    # Configura o Handler para saída no terminal (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # Pega o logger raiz (root)
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Limpa handlers anteriores para evitar logs duplicados se a função for chamada 2x
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
        
    root_logger.addHandler(console_handler)

    # DICA DE OURO: Silencia logs chatos de bibliotecas externas (opcional)
    # Isso evita que o 'httpx' (usado pela LLM) polua seu terminal com requests HTTP
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    return root_logger