"""CLI commands for claude-code-model-gateway."""

import logging
from pathlib import Path
from typing import Optional

import click

from src import __version__
from src.logging_config import (
    LogFormat,
    LoggingConfig,
    get_log_files,
    get_logger,
    get_logging_status,
    set_log_level,
    setup_logging,
)


@click.group()
@click.version_option(version=__version__, prog_name="claude-code-model-gateway")
def main():
    """claude-code-model-gateway - A command-line application.

    Use --help on any command for more information.
    """
    pass


@main.command()
def hello():
    """Say hello from claude-code-model-gateway."""
    click.echo("Hello from claude-code-model-gateway!")
    click.echo("Use --help for available commands.")


@main.command()
@click.argument("name", default="World")
def greet(name: str):
    """Greet someone by NAME.

    If no name is provided, greets "World".
    """
    click.echo(f"Hello, {name}!")


@main.command()
def version():
    """Show the application version."""
    click.echo(f"claude-code-model-gateway version {__version__}")


@main.command()
@click.option(
    "--host",
    "-H",
    default="127.0.0.1",
    show_default=True,
    help="Host address to bind to.",
)
@click.option(
    "--port",
    "-p",
    default=3000,
    show_default=True,
    type=int,
    help="Port to listen on.",
)
@click.option(
    "--timeout",
    "-t",
    default=30,
    show_default=True,
    type=int,
    help="Upstream connection timeout in seconds.",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Enable verbose (DEBUG) logging.",
)
@click.option(
    "--log-format",
    type=click.Choice(["standard", "detailed", "json", "colored", "minimal"]),
    default="standard",
    show_default=True,
    help="Log output format.",
)
@click.option(
    "--log-file",
    default=None,
    type=click.Path(dir_okay=False),
    help="Path to log file (enables file logging).",
)
@click.option(
    "--max-retries",
    default=0,
    show_default=True,
    type=int,
    help="Number of retries for failed upstream requests (0 = no retries).",
)
@click.option(
    "--retry-delay",
    default=1.0,
    show_default=True,
    type=float,
    help="Base delay in seconds between retries (exponential backoff).",
)
def proxy(
    host: str,
    port: int,
    timeout: int,
    verbose: bool,
    log_format: str,
    log_file: str,
    max_retries: int,
    retry_delay: float,
):
    """Start the HTTP proxy server.

    Launches a forward HTTP proxy on the specified HOST:PORT.
    Incoming requests are forwarded to the target server and responses
    are relayed back to the client.

    Retry logic applies exponential backoff with jitter for transient
    upstream failures (connection errors, 5xx responses).

    Example:

        claude-code-model-gateway proxy --host 127.0.0.1 --port 3000

        claude-code-model-gateway proxy --max-retries 3 --retry-delay 0.5
    """
    from src.proxy import run_proxy

    log_level = "debug" if verbose else "info"
    output = "both" if log_file else "console"
    setup_logging(
        level=log_level,
        log_format=log_format,
        output=output,
        log_file=log_file,
    )

    logger = get_logger("cli.proxy")
    logger.info("Starting HTTP proxy server on %s:%d", host, port)
    click.echo(f"Starting HTTP proxy server on {host}:{port} ...")
    if max_retries > 0:
        click.echo(f"  Retries:   {max_retries} (base delay: {retry_delay}s)")
    run_proxy(
        host=host,
        port=port,
        timeout=timeout,
        max_retries=max_retries,
        retry_base_delay=retry_delay,
    )


@main.command("gateway")
@click.option(
    "--host",
    "-H",
    default="127.0.0.1",
    show_default=True,
    help="Host address to bind to.",
)
@click.option(
    "--port",
    "-p",
    default=8080,
    show_default=True,
    type=int,
    help="Port to listen on.",
)
@click.option(
    "--timeout",
    "-t",
    default=300,
    show_default=True,
    type=int,
    help="Upstream connection timeout in seconds.",
)
@click.option(
    "--api-key",
    default=None,
    envvar="ANTHROPIC_API_KEY",
    help="Anthropic API key (defaults to ANTHROPIC_API_KEY env var).",
)
@click.option(
    "--anthropic-version",
    default="2023-06-01",
    show_default=True,
    help="Anthropic API version header value.",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Enable verbose (DEBUG) logging.",
)
@click.option(
    "--log-format",
    type=click.Choice(["standard", "detailed", "json", "colored", "minimal"]),
    default="standard",
    show_default=True,
    help="Log output format.",
)
@click.option(
    "--log-file",
    default=None,
    type=click.Path(dir_okay=False),
    help="Path to log file (enables file logging).",
)
@click.option(
    "--max-retries",
    default=2,
    show_default=True,
    type=int,
    help="Number of retries for failed non-streaming requests (0 = no retries).",
)
@click.option(
    "--retry-delay",
    default=1.0,
    show_default=True,
    type=float,
    help="Base delay in seconds between retries (exponential backoff).",
)
@click.option(
    "--passthrough-mode",
    is_flag=True,
    default=False,
    help="Enable pass-through mode: forward all /v1/* paths (not just whitelisted).",
)
@click.option(
    "--max-request-size",
    default=10485760,
    show_default=True,
    type=int,
    help="Maximum request body size in bytes (0 = unlimited).",
)
@click.option(
    "--no-connection-pool",
    is_flag=True,
    default=False,
    help="Disable connection pooling to upstream.",
)
@click.option(
    "--pool-size",
    default=10,
    show_default=True,
    type=int,
    help="Maximum number of pooled connections to upstream.",
)
@click.option(
    "--response-cache/--no-response-cache",
    default=False,
    show_default=True,
    help=(
        "Enable response caching for GET/HEAD requests (e.g. /v1/models). "
        "Reduces upstream calls for read-only endpoints."
    ),
)
@click.option(
    "--cache-ttl",
    default=300,
    show_default=True,
    type=int,
    help="Time-to-live in seconds for cached responses (used with --response-cache).",
)
@click.option(
    "--cache-maxsize",
    default=256,
    show_default=True,
    type=int,
    help="Maximum number of responses held in the response cache.",
)
@click.option(
    "--token-count-cache/--no-token-count-cache",
    default=False,
    show_default=True,
    help=(
        "Enable token count caching for POST /v1/messages/count_tokens. "
        "Token counting is deterministic, so identical requests are served "
        "from cache instead of hitting the Anthropic API."
    ),
)
@click.option(
    "--token-count-cache-ttl",
    default=3600,
    show_default=True,
    type=int,
    help=(
        "Time-to-live in seconds for cached token count responses "
        "(used with --token-count-cache). Default: 1 hour."
    ),
)
@click.option(
    "--token-count-cache-maxsize",
    default=512,
    show_default=True,
    type=int,
    help="Maximum number of token count responses held in the token count cache.",
)
def gateway(
    host: str,
    port: int,
    timeout: int,
    api_key: Optional[str],
    anthropic_version: str,
    verbose: bool,
    log_format: str,
    log_file: str,
    max_retries: int,
    retry_delay: float,
    passthrough_mode: bool,
    max_request_size: int,
    no_connection_pool: bool,
    pool_size: int,
    response_cache: bool,
    cache_ttl: int,
    cache_maxsize: int,
    token_count_cache: bool,
    token_count_cache_ttl: int,
    token_count_cache_maxsize: int,
):
    """Start the Anthropic API pass-through gateway.

    Launches a local HTTP server that acts as a transparent reverse proxy
    to the Anthropic Messages API. All requests are forwarded directly to
    api.anthropic.com with proper authentication headers.

    Supported endpoints (strict mode):

    \b
      POST /v1/messages              - Create a message
      POST /v1/messages/count_tokens - Count tokens
      POST /v1/messages/batches      - Create message batch
      GET  /v1/messages/batches      - List message batches
      GET  /v1/messages/batches/{id} - Get batch status/results
      GET  /v1/models                - List available models
      GET  /v1/models/{id}           - Get model details

    In --passthrough-mode, ALL /v1/* paths are forwarded transparently.

    Internal gateway endpoints (never forwarded):

    \b
      GET  /health                   - Health check
      GET  /status                   - Gateway metrics and status

    The gateway reads the API key from --api-key, the client's x-api-key
    header, or the ANTHROPIC_API_KEY environment variable (in that order).

    Both streaming (SSE) and non-streaming responses are fully supported.
    Non-streaming requests are automatically retried on transient failures
    (429, 5xx) with exponential backoff.  Streaming requests are NOT
    retried because partial SSE data may have already been sent.

    Features:

    \b
      - Request correlation IDs (auto-generated or forwarded x-request-id)
      - Connection pooling for improved performance
      - Optional response caching for read-only GET endpoints
      - Request body size validation
      - Health check and metrics endpoints
      - Full HTTP method support (GET, POST, PUT, DELETE, PATCH, HEAD)

    Example:

    \b
        claude-code-model-gateway gateway
        claude-code-model-gateway gateway --port 9090
        claude-code-model-gateway gateway --api-key sk-ant-...
        claude-code-model-gateway gateway -v --timeout 600
        claude-code-model-gateway gateway --max-retries 5 --retry-delay 2.0
        claude-code-model-gateway gateway --passthrough-mode
        claude-code-model-gateway gateway --response-cache --cache-ttl 600
    """
    from src.anthropic_passthrough import run_passthrough

    log_level = "debug" if verbose else "info"
    output = "both" if log_file else "console"
    setup_logging(
        level=log_level,
        log_format=log_format,
        output=output,
        log_file=log_file,
    )

    logger = get_logger("cli.gateway")

    if not api_key:
        import os

        api_key = os.environ.get("ANTHROPIC_API_KEY")

    mode = "pass-through" if passthrough_mode else "strict"
    key_source = "provided" if api_key else "not set (will check per-request)"
    pool_status = "disabled" if no_connection_pool else f"enabled (max {pool_size})"
    cache_status = (
        f"enabled (ttl={cache_ttl}s, maxsize={cache_maxsize})"
        if response_cache
        else "disabled"
    )
    token_count_cache_status = (
        f"enabled (ttl={token_count_cache_ttl}s, maxsize={token_count_cache_maxsize})"
        if token_count_cache
        else "disabled"
    )

    click.echo(f"Starting Anthropic pass-through gateway on {host}:{port}")
    click.echo(f"  Upstream:  https://api.anthropic.com/v1")
    click.echo(f"  API key:   {key_source}")
    click.echo(f"  Version:   {anthropic_version}")
    click.echo(f"  Mode:      {mode}")
    click.echo(f"  Timeout:   {timeout}s")
    click.echo(f"  Retries:   {max_retries} (base delay: {retry_delay}s)")
    click.echo(f"  Pool:      {pool_status}")
    click.echo(f"  Cache:     {cache_status}")
    click.echo(f"  TC Cache:  {token_count_cache_status}")
    if max_request_size > 0:
        size_mb = max_request_size / (1024 * 1024)
        click.echo(f"  Max body:  {size_mb:.1f} MB")
    else:
        click.echo(f"  Max body:  unlimited")
    click.echo()

    if passthrough_mode:
        click.echo("Pass-through mode: ALL /v1/* paths forwarded to Anthropic API")
    else:
        click.echo("Endpoints:")
        click.echo(f"  POST http://{host}:{port}/v1/messages")
        click.echo(f"  POST http://{host}:{port}/v1/messages/count_tokens")
        click.echo(f"  POST http://{host}:{port}/v1/messages/batches")
        click.echo(f"  GET  http://{host}:{port}/v1/models")

    click.echo()
    click.echo("Internal endpoints:")
    click.echo(f"  GET  http://{host}:{port}/health")
    click.echo(f"  GET  http://{host}:{port}/status")
    click.echo()

    run_passthrough(
        host=host,
        port=port,
        timeout=timeout,
        api_key=api_key,
        anthropic_version=anthropic_version,
        max_retries=max_retries,
        retry_base_delay=retry_delay,
        passthrough_mode=passthrough_mode,
        max_request_size=max_request_size,
        enable_connection_pool=not no_connection_pool,
        max_pool_connections=pool_size,
        enable_response_cache=response_cache,
        response_cache_ttl=float(cache_ttl),
        response_cache_maxsize=cache_maxsize,
        enable_token_count_cache=token_count_cache,
        token_count_cache_ttl=float(token_count_cache_ttl),
        token_count_cache_maxsize=token_count_cache_maxsize,
    )


# ---------------------------------------------------------------------------
# Configuration commands
# ---------------------------------------------------------------------------


@main.group()
def config():
    """Manage gateway configuration.

    View, initialize, and modify the model gateway configuration file.
    """
    pass


@config.command("init")
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False),
    default="gateway.yaml",
    show_default=True,
    help="Output path for the configuration file.",
)
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["yaml", "json"]),
    default="yaml",
    show_default=True,
    help="Configuration file format.",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Overwrite existing configuration file.",
)
def config_init(output: str, fmt: str, force: bool):
    """Initialize a new configuration file with default providers.

    Creates a configuration file pre-populated with all built-in provider
    templates (OpenAI, Anthropic, Azure, Google, AWS Bedrock).

    Example:

        claude-code-model-gateway config init

        claude-code-model-gateway config init -o myconfig.json --format json
    """
    from src.config import ConfigError, get_default_config, save_config_file

    # Ensure correct extension
    path = Path(output)
    if fmt == "json" and path.suffix not in (".json",):
        path = path.with_suffix(".json")
    elif fmt == "yaml" and path.suffix not in (".yaml", ".yml"):
        path = path.with_suffix(".yaml")

    if path.exists() and not force:
        click.echo(f"Error: Configuration file '{path}' already exists.", err=True)
        click.echo("Use --force to overwrite.", err=True)
        raise SystemExit(1)

    try:
        config_obj = get_default_config()
        save_config_file(config_obj, path)
        click.echo(f"Configuration file created: {path}")
        click.echo(
            f"  {len(config_obj.providers)} providers configured "
            f"(default: {config_obj.default_provider})"
        )
    except ConfigError as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)


@config.command("show")
@click.option(
    "--config-file",
    "-c",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Path to configuration file.",
)
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["yaml", "json", "text"]),
    default="text",
    show_default=True,
    help="Output format.",
)
def config_show(config_file: Optional[str], fmt: str):
    """Display the current configuration.

    Reads the configuration from file (auto-detected or specified)
    and displays it in the requested format.

    Example:

        claude-code-model-gateway config show

        claude-code-model-gateway config show --format yaml
    """
    import json as json_mod

    import yaml

    from src.config import ConfigError, ConfigValidationError, load_config

    path = Path(config_file) if config_file else None

    try:
        config_obj = load_config(path=path, validate=False)
    except ConfigError as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)

    data = config_obj.to_dict()

    if fmt == "yaml":
        click.echo(yaml.dump(data, default_flow_style=False, sort_keys=False))
    elif fmt == "json":
        click.echo(json_mod.dumps(data, indent=2))
    else:
        _print_config_text(config_obj)


def _print_config_text(config_obj):
    """Pretty-print configuration in a human-readable text format."""
    from src.config import validate_config

    click.echo("Gateway Configuration")
    click.echo("=" * 50)
    click.echo(f"  Default provider: {config_obj.default_provider or '(none)'}")
    click.echo(f"  Log level:        {config_obj.log_level}")
    click.echo(f"  Timeout:          {config_obj.timeout}s")
    click.echo(f"  Max retries:      {config_obj.max_retries}")
    click.echo()

    if config_obj.providers:
        click.echo(f"Providers ({len(config_obj.providers)}):")
        click.echo("-" * 50)
        for name, provider in sorted(config_obj.providers.items()):
            status = click.style("enabled", fg="green") if provider.enabled else click.style("disabled", fg="red")
            default_marker = " *" if name == config_obj.default_provider else ""
            click.echo(f"  {provider.display_name} ({name}){default_marker} [{status}]")
            click.echo(f"    API base:     {provider.api_base}")
            click.echo(f"    Auth:         {provider.auth_type.value}")
            if provider.api_key_env_var:
                click.echo(f"    API key env:  ${provider.api_key_env_var}")
            click.echo(f"    Default model: {provider.default_model or '(none)'}")
            if provider.models:
                click.echo(f"    Models ({len(provider.models)}):")
                for model_name, model in sorted(provider.models.items()):
                    features = []
                    if model.supports_streaming:
                        features.append("stream")
                    if model.supports_tools:
                        features.append("tools")
                    if model.supports_vision:
                        features.append("vision")
                    feature_str = ", ".join(features) if features else "basic"
                    click.echo(f"      - {model.display_name} ({model_name}) [{feature_str}]")
            click.echo()
    else:
        click.echo("No providers configured.")
        click.echo()

    # Run validation
    errors = validate_config(config_obj)
    if errors:
        click.echo(click.style("Validation warnings:", fg="yellow"))
        for error in errors:
            click.echo(f"  ⚠ {error}")


@config.command("validate")
@click.option(
    "--config-file",
    "-c",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Path to configuration file.",
)
def config_validate(config_file: Optional[str]):
    """Validate the configuration file.

    Checks for common issues such as missing fields, invalid references,
    and configuration inconsistencies.

    Example:

        claude-code-model-gateway config validate

        claude-code-model-gateway config validate -c gateway.yaml
    """
    from src.config import ConfigError, load_config, validate_config

    path = Path(config_file) if config_file else None

    try:
        config_obj = load_config(path=path, validate=False)
    except ConfigError as e:
        click.echo(f"Error loading config: {e}", err=True)
        raise SystemExit(1)

    errors = validate_config(config_obj)
    if errors:
        click.echo(click.style("Configuration validation failed:", fg="red"))
        for error in errors:
            click.echo(f"  ✗ {error}")
        raise SystemExit(1)
    else:
        click.echo(click.style("Configuration is valid.", fg="green"))


@config.command("set")
@click.argument("key")
@click.argument("value")
@click.option(
    "--config-file",
    "-c",
    type=click.Path(dir_okay=False),
    default=None,
    help="Path to configuration file.",
)
def config_set(key: str, value: str, config_file: Optional[str]):
    """Set a top-level configuration value.

    Supported keys: default_provider, log_level, timeout, max_retries.

    Example:

        claude-code-model-gateway config set default_provider anthropic

        claude-code-model-gateway config set timeout 60
    """
    from src.config import (
        ConfigError,
        find_config_file,
        load_config,
        save_config_file,
    )

    path = Path(config_file) if config_file else find_config_file()
    if path is None:
        click.echo(
            "Error: No configuration file found. Run 'config init' first.",
            err=True,
        )
        raise SystemExit(1)

    try:
        config_obj = load_config(path=path, validate=False)
    except ConfigError as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)

    valid_keys = {"default_provider", "log_level", "timeout", "max_retries"}
    if key not in valid_keys:
        click.echo(
            f"Error: Unknown key '{key}'. Valid keys: {', '.join(sorted(valid_keys))}",
            err=True,
        )
        raise SystemExit(1)

    if key == "timeout":
        try:
            setattr(config_obj, key, int(value))
        except ValueError:
            click.echo(f"Error: '{value}' is not a valid integer.", err=True)
            raise SystemExit(1)
    elif key == "max_retries":
        try:
            setattr(config_obj, key, int(value))
        except ValueError:
            click.echo(f"Error: '{value}' is not a valid integer.", err=True)
            raise SystemExit(1)
    else:
        setattr(config_obj, key, value)

    try:
        save_config_file(config_obj, path)
        click.echo(f"Set {key} = {value}")
    except ConfigError as e:
        click.echo(f"Error saving config: {e}", err=True)
        raise SystemExit(1)


@config.command("schema")
@click.option(
    "--field",
    "-f",
    default=None,
    help="Show details for a specific field path (e.g., 'timeout', 'providers.*.api_base').",
)
@click.option(
    "--category",
    type=click.Choice(["gateway", "provider", "model", "logging", "retry", "proxy"]),
    default=None,
    help="Filter fields by category.",
)
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["text", "json", "template"]),
    default="text",
    show_default=True,
    help="Output format: text (human readable), json (machine readable), template (YAML template).",
)
def config_schema(field: Optional[str], category: Optional[str], fmt: str):
    """Show configuration schema documentation.

    Displays all available configuration fields with their types,
    defaults, constraints, and descriptions.  Use --field to inspect
    a single field in detail or --category to filter by group.

    Use --format template to generate a documented YAML template
    you can use as a starting point for your configuration.

    Example:

        claude-code-model-gateway config schema

        claude-code-model-gateway config schema --field timeout

        claude-code-model-gateway config schema --category provider

        claude-code-model-gateway config schema --format template > gateway.yaml
    """
    import json as json_mod

    from src.config.schema import (
        FieldCategory,
        generate_documented_config,
        get_field,
        list_fields,
    )

    if fmt == "template":
        click.echo(generate_documented_config())
        return

    if field:
        # Show a single field
        schema_field = get_field(field)
        if schema_field is None:
            click.secho(f"Error: Unknown field path '{field}'.", fg="red", err=True)
            click.echo(
                "Use 'config schema' without --field to see all available fields.",
                err=True,
            )
            raise SystemExit(1)

        if fmt == "json":
            click.echo(json_mod.dumps(schema_field.to_dict(), indent=2))
            return

        # Text format for single field
        click.echo(f"Field: {field}")
        click.echo("=" * 60)
        click.echo(f"  Type:        {schema_field.field_type.value}")
        click.echo(f"  Category:    {schema_field.category.value}")
        click.echo(f"  Description: {schema_field.description}")
        if schema_field.default is not None:
            click.echo(f"  Default:     {schema_field.default!r}")
        if schema_field.env_var:
            click.echo(f"  Env var:     {schema_field.env_var}")
        if schema_field.constraint.required:
            click.echo("  Required:    yes")
        if schema_field.constraint.min_value is not None:
            click.echo(f"  Min value:   {schema_field.constraint.min_value}")
        if schema_field.constraint.max_value is not None:
            click.echo(f"  Max value:   {schema_field.constraint.max_value}")
        if schema_field.constraint.min_length is not None:
            click.echo(f"  Min length:  {schema_field.constraint.min_length}")
        if schema_field.constraint.max_length is not None:
            click.echo(f"  Max length:  {schema_field.constraint.max_length}")
        if schema_field.constraint.pattern:
            click.echo(f"  Pattern:     {schema_field.constraint.pattern}")
        if schema_field.constraint.allowed_values:
            click.echo(
                f"  Allowed:     {', '.join(schema_field.constraint.allowed_values)}"
            )
        if schema_field.examples:
            click.echo(f"  Examples:    {', '.join(str(e) for e in schema_field.examples)}")
        if schema_field.deprecated:
            click.secho(
                f"  DEPRECATED:  {schema_field.deprecation_message or 'This field is deprecated.'}",
                fg="yellow",
            )
        if schema_field.children:
            child_names = [
                k for k in schema_field.children if k != "*"
            ]
            if child_names:
                click.echo(f"  Children:    {', '.join(sorted(child_names))}")
        return

    # List all fields
    cat_filter = None
    if category:
        try:
            cat_filter = FieldCategory(category)
        except ValueError:
            click.secho(f"Error: Unknown category '{category}'.", fg="red", err=True)
            raise SystemExit(1)

    fields = list_fields(category=cat_filter, include_nested=True)
    # Filter out wildcard schema entries
    fields = [f for f in fields if f.name != "*"]

    if fmt == "json":
        click.echo(
            json_mod.dumps(
                {"fields": [f.to_dict() for f in fields], "total": len(fields)},
                indent=2,
            )
        )
        return

    title = "Configuration Schema"
    if category:
        title += f" (category: {category})"
    click.echo(title)
    click.echo("=" * 60)

    # Group by category
    grouped: dict[str, list] = {}
    for f in fields:
        cat = f.category.value
        grouped.setdefault(cat, []).append(f)

    for cat_name, cat_fields in sorted(grouped.items()):
        click.echo(f"\n{cat_name.upper()} FIELDS:")
        click.echo("-" * 40)
        for f in cat_fields:
            type_str = f.field_type.value
            default_str = f" (default: {f.default!r})" if f.default is not None else ""
            required_str = " [required]" if f.constraint.required else ""
            env_str = f" [env: {f.env_var}]" if f.env_var else ""
            deprecated_str = " [DEPRECATED]" if f.deprecated else ""
            click.echo(
                f"  {f.name:<30} {type_str:<10}{required_str}{default_str}{env_str}{deprecated_str}"
            )
            if f.description:
                click.echo(f"    {f.description}")


@config.command("lint")
@click.option(
    "--config-file",
    "-c",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Path to configuration file. Auto-discovered if not specified.",
)
@click.option(
    "--profile",
    type=click.Choice(["relaxed", "default", "strict", "production"]),
    default="default",
    show_default=True,
    help="Validation profile to apply.",
)
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["text", "json"]),
    default="text",
    show_default=True,
    help="Output format.",
)
@click.option(
    "--no-suggestions",
    is_flag=True,
    default=False,
    help="Suppress fix suggestions in output.",
)
def config_lint(
    config_file: Optional[str],
    profile: str,
    fmt: str,
    no_suggestions: bool,
):
    """Lint the configuration file for issues and best-practice violations.

    Produces actionable, lint-style output with file path, field path,
    severity, message, and (optionally) suggested fixes.

    Exit codes:
      0  No issues found
      1  One or more errors found
      2  One or more warnings found (no errors)

    Example:

        claude-code-model-gateway config lint

        claude-code-model-gateway config lint -c gateway.yaml --profile strict

        claude-code-model-gateway config lint --format json
    """
    import json as json_mod

    from src.config import ConfigError, load_config
    from src.config.validator import full_validate

    path = Path(config_file) if config_file else None

    try:
        config_obj = load_config(path=path, validate=False)
    except ConfigError as e:
        if fmt == "json":
            click.echo(
                json_mod.dumps({"file": str(path or "auto"), "error": str(e), "messages": []}, indent=2)
            )
        else:
            click.secho(f"Error: {e}", fg="red", err=True)
        raise SystemExit(1)

    result = full_validate(config_obj, profile=profile)

    # Determine display path
    from src.config import find_config_file
    resolved_path = path or find_config_file()
    file_label = str(resolved_path) if resolved_path else "<config>"

    if fmt == "json":
        output = {
            "file": file_label,
            "profile": profile,
            "error_count": result.error_count,
            "warning_count": result.warning_count,
            "info_count": len(result.infos),
            "messages": [msg.to_dict() for msg in result.messages],
        }
        click.echo(json_mod.dumps(output, indent=2))
        if result.error_count > 0:
            raise SystemExit(1)
        elif result.warning_count > 0:
            raise SystemExit(2)
        raise SystemExit(0)

    # Text lint output (ESLint-style)
    if result.messages:
        click.echo(f"\n{file_label}")
        click.echo()
        for msg in result.messages:
            sev = msg.severity.value
            if sev == "error":
                sev_colored = click.style("error  ", fg="red", bold=True)
            elif sev == "warning":
                sev_colored = click.style("warning", fg="yellow")
            else:
                sev_colored = click.style("info   ", fg="cyan")

            field_str = click.style(f"  {msg.path:<45}", fg="white")
            click.echo(f"{field_str} {sev_colored}  {msg.message}")
            if not no_suggestions and msg.suggestion:
                click.echo(f"    {'':45}  ↳ {msg.suggestion}")
        click.echo()
    else:
        click.secho(f"  {file_label}", fg="white")
        click.echo()

    # Summary line
    errors = result.error_count
    warnings = result.warning_count
    infos = len(result.infos)

    summary_parts = []
    if errors:
        summary_parts.append(click.style(f"✗ {errors} error{'s' if errors != 1 else ''}", fg="red", bold=True))
    if warnings:
        summary_parts.append(click.style(f"⚠ {warnings} warning{'s' if warnings != 1 else ''}", fg="yellow"))
    if infos:
        summary_parts.append(click.style(f"ℹ {infos} info", fg="cyan"))

    if summary_parts:
        click.echo("  " + "  ".join(summary_parts))
    else:
        click.secho("  ✓ No issues found.", fg="green")

    if result.error_count > 0:
        raise SystemExit(1)
    elif result.warning_count > 0:
        raise SystemExit(2)
    raise SystemExit(0)


@config.command("doctor")
@click.option(
    "--config-file",
    "-c",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Path to configuration file. Auto-discovered if not specified.",
)
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["text", "json"]),
    default="text",
    show_default=True,
    help="Output format.",
)
@click.option(
    "--check-env",
    is_flag=True,
    default=False,
    help="Check that required environment variables are set.",
)
def config_doctor(config_file: Optional[str], fmt: str, check_env: bool):
    """Run a comprehensive configuration health check.

    Executes a series of diagnostic checks against the configuration
    and reports issues with clear remediation guidance:

    \b
      1. Parse check     - Can the file be loaded?
      2. Schema check    - Do all fields pass type and constraint validation?
      3. Semantic check  - Are cross-field references and values consistent?
      4. Production check- Does the config meet production-readiness standards?
      5. Env var check   - Are required environment variables set? (--check-env)

    Exit codes: 0 = healthy, 1 = one or more checks failed.

    Example:

        claude-code-model-gateway config doctor

        claude-code-model-gateway config doctor --check-env

        claude-code-model-gateway config doctor -c prod.yaml --format json
    """
    import json as json_mod
    import os

    from src.config import ConfigError, load_config_file
    from src.config.schema import validate_against_schema
    from src.config.validator import full_validate, validate_for_production
    from src.models import GatewayConfig, AuthType

    path = Path(config_file) if config_file else None
    if path is None:
        from src.config import find_config_file
        path = find_config_file()

    checks = []  # list of {"name": ..., "passed": ..., "message": ..., "details": [...]}

    # ---- Check 1: Parse ----
    raw_data: dict = {}
    parse_passed = True
    parse_message = ""
    try:
        if path is None:
            raw_data = {}
            parse_message = "No config file found; using empty defaults."
        else:
            raw_data = load_config_file(path)
            parse_message = f"Loaded {path}"
    except ConfigError as e:
        parse_passed = False
        parse_message = str(e)
    checks.append(
        {
            "name": "parse",
            "description": "Configuration file can be loaded and parsed",
            "passed": parse_passed,
            "message": parse_message,
            "details": [],
        }
    )

    if not parse_passed:
        _emit_doctor_report(checks, fmt, path)
        raise SystemExit(1)

    # ---- Check 2: Schema ----
    schema_errors = validate_against_schema(raw_data)
    schema_passed = len(schema_errors) == 0
    checks.append(
        {
            "name": "schema",
            "description": "All fields pass type and constraint validation",
            "passed": schema_passed,
            "message": "Schema is valid." if schema_passed else f"{len(schema_errors)} schema error(s).",
            "details": schema_errors,
        }
    )

    # ---- Check 3: Semantic ----
    try:
        config_obj = GatewayConfig.from_dict(raw_data)
    except Exception as e:
        checks.append(
            {
                "name": "semantic",
                "description": "Cross-field references and values are consistent",
                "passed": False,
                "message": f"Could not build config model: {e}",
                "details": [],
            }
        )
        _emit_doctor_report(checks, fmt, path)
        raise SystemExit(1)

    sem_result = full_validate(config_obj, profile="default")
    sem_passed = sem_result.is_valid
    sem_details = [str(m) for m in sem_result.errors + sem_result.warnings]
    checks.append(
        {
            "name": "semantic",
            "description": "Cross-field references and values are consistent",
            "passed": sem_passed,
            "message": (
                "Semantic checks passed."
                if sem_passed
                else f"{sem_result.error_count} error(s), {sem_result.warning_count} warning(s)."
            ),
            "details": sem_details,
        }
    )

    # ---- Check 4: Production readiness ----
    prod_ok, prod_report = validate_for_production(raw_data)
    prod_details = [
        line.strip()
        for line in prod_report.splitlines()
        if line.strip() and not line.startswith("Configuration is")
    ]
    checks.append(
        {
            "name": "production",
            "description": "Configuration meets production-readiness standards",
            "passed": prod_ok,
            "message": "Production checks passed." if prod_ok else "Production checks failed.",
            "details": prod_details,
        }
    )

    # ---- Check 5: Environment variables (optional) ----
    if check_env:
        env_details = []
        env_passed = True
        for pname, provider in config_obj.providers.items():
            if not provider.enabled:
                continue
            if provider.auth_type == AuthType.NONE:
                continue
            env_var = provider.api_key_env_var
            if not env_var:
                env_details.append(f"providers.{pname}: No api_key_env_var configured.")
                env_passed = False
            elif not os.environ.get(env_var):
                env_details.append(
                    f"providers.{pname}: ${env_var} is not set in the environment."
                )
                env_passed = False
            else:
                env_details.append(f"providers.{pname}: ${env_var} ✓")
        if not config_obj.providers:
            env_details.append("No providers configured — nothing to check.")
        checks.append(
            {
                "name": "env_vars",
                "description": "Required environment variables are set",
                "passed": env_passed,
                "message": (
                    "All required env vars are set."
                    if env_passed
                    else "Some required env vars are missing."
                ),
                "details": env_details,
            }
        )

    _emit_doctor_report(checks, fmt, path)
    all_passed = all(c["passed"] for c in checks)
    raise SystemExit(0 if all_passed else 1)


def _emit_doctor_report(
    checks: list,
    fmt: str,
    path,
) -> None:
    """Emit the doctor report in the requested format."""
    import json as json_mod

    if fmt == "json":
        click.echo(
            json_mod.dumps(
                {
                    "config_file": str(path) if path else None,
                    "all_passed": all(c["passed"] for c in checks),
                    "checks": checks,
                },
                indent=2,
            )
        )
        return

    file_str = str(path) if path else "(no file)"
    click.echo(f"\nConfiguration Doctor Report: {file_str}")
    click.echo("=" * 60)
    click.echo()

    all_passed = True
    for check in checks:
        icon = "✓" if check["passed"] else "✗"
        color = "green" if check["passed"] else "red"
        status = click.style(f"[{icon}]", fg=color, bold=True)
        click.echo(f"  {status} {check['name'].upper()}: {check['description']}")
        click.echo(f"       {check['message']}")
        for detail in check["details"]:
            if detail.startswith("[ERROR]") or "error" in detail.lower():
                click.echo(f"       {click.style('↳', fg='red')} {detail}")
            elif detail.startswith("[WARNING]") or "warning" in detail.lower():
                click.echo(f"       {click.style('↳', fg='yellow')} {detail}")
            elif "✓" in detail:
                click.echo(f"       {click.style('↳', fg='green')} {detail}")
            else:
                click.echo(f"       ↳ {detail}")
        click.echo()
        if not check["passed"]:
            all_passed = False

    if all_passed:
        click.secho("  ✓ All checks passed. Configuration is healthy.", fg="green", bold=True)
    else:
        failed_names = [c["name"] for c in checks if not c["passed"]]
        click.secho(
            f"  ✗ {len(failed_names)} check(s) failed: {', '.join(failed_names)}",
            fg="red",
            bold=True,
        )
    click.echo()


@config.command("export")
@click.option(
    "--config-file",
    "-c",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Source configuration file. Auto-discovered if not specified.",
)
@click.option(
    "--output",
    "-o",
    required=True,
    type=click.Path(dir_okay=False),
    help="Output file path (.yaml, .yml, or .json).",
)
@click.option(
    "--validate/--no-validate",
    default=True,
    show_default=True,
    help="Validate the configuration before exporting.",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Overwrite output file if it exists.",
)
def config_export(
    config_file: Optional[str],
    output: str,
    validate: bool,
    force: bool,
):
    """Export configuration to a different format.

    Reads the current configuration and writes it to the specified
    output file.  The format is determined by the file extension
    (.yaml/.yml for YAML, .json for JSON).

    Useful for converting between YAML and JSON, or for making a
    copy of the current configuration in a different location.

    Example:

        claude-code-model-gateway config export -o backup.yaml

        claude-code-model-gateway config export -o config.json

        claude-code-model-gateway config export -c dev.yaml -o prod.yaml --no-validate
    """
    from src.config import ConfigError, load_config, save_config_file

    src_path = Path(config_file) if config_file else None
    dst_path = Path(output)

    suffix = dst_path.suffix.lower()
    if suffix not in (".yaml", ".yml", ".json"):
        click.secho(
            f"Error: Unsupported output format '{suffix}'. Use .yaml, .yml, or .json.",
            fg="red",
            err=True,
        )
        raise SystemExit(1)

    if dst_path.exists() and not force:
        click.echo(f"Error: '{dst_path}' already exists. Use --force to overwrite.", err=True)
        raise SystemExit(1)

    try:
        config_obj = load_config(path=src_path, validate=validate)
    except ConfigError as e:
        click.secho(f"Error: {e}", fg="red", err=True)
        raise SystemExit(1)

    try:
        save_config_file(config_obj, dst_path)
    except ConfigError as e:
        click.secho(f"Error writing config: {e}", fg="red", err=True)
        raise SystemExit(1)

    src_label = str(src_path) if src_path else "(auto-discovered)"
    click.echo(f"Exported: {src_label} → {dst_path}")
    click.echo(f"  Format:    {suffix.lstrip('.')}")
    click.echo(f"  Providers: {len(config_obj.providers)}")


@config.command("env-check")
@click.option(
    "--config-file",
    "-c",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Path to configuration file. Auto-discovered if not specified.",
)
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["text", "json"]),
    default="text",
    show_default=True,
    help="Output format.",
)
@click.option(
    "--all",
    "show_all",
    is_flag=True,
    default=False,
    help="Show all env vars including those that are already set.",
)
def config_env_check(config_file: Optional[str], fmt: str, show_all: bool):
    """Check that required environment variables are configured.

    Reads the configuration, identifies all environment variables
    referenced by providers (api_key_env_var fields), and reports
    which ones are present and which are missing.

    Exit codes: 0 = all set, 1 = one or more missing.

    Example:

        claude-code-model-gateway config env-check

        claude-code-model-gateway config env-check --all

        claude-code-model-gateway config env-check --format json
    """
    import json as json_mod
    import os

    from src.config import ConfigError, load_config
    from src.models import AuthType

    path = Path(config_file) if config_file else None

    try:
        config_obj = load_config(path=path, validate=False)
    except ConfigError as e:
        if fmt == "json":
            click.echo(json_mod.dumps({"error": str(e), "env_vars": []}, indent=2))
        else:
            click.secho(f"Error: {e}", fg="red", err=True)
        raise SystemExit(1)

    env_entries = []
    for pname, provider in sorted(config_obj.providers.items()):
        if not provider.enabled:
            status = "disabled"
            present = None
            env_var = provider.api_key_env_var or "(none)"
        elif provider.auth_type == AuthType.NONE:
            status = "no-auth"
            present = None
            env_var = "(no auth required)"
        elif not provider.api_key_env_var:
            status = "missing-config"
            present = False
            env_var = "(not configured)"
        else:
            env_var = provider.api_key_env_var
            val = os.environ.get(env_var)
            present = val is not None
            status = "set" if present else "missing"

        env_entries.append(
            {
                "provider": pname,
                "env_var": env_var,
                "status": status,
                "present": present,
            }
        )

    if fmt == "json":
        missing = [e for e in env_entries if e["status"] in ("missing", "missing-config")]
        click.echo(
            json_mod.dumps(
                {
                    "env_vars": env_entries,
                    "missing_count": len(missing),
                    "all_set": len(missing) == 0,
                },
                indent=2,
            )
        )
        all_set = all(e["status"] not in ("missing", "missing-config") for e in env_entries)
        raise SystemExit(0 if all_set else 1)

    # Text output
    if not config_obj.providers:
        click.echo("No providers configured.")
        raise SystemExit(0)

    click.echo("Environment Variable Check")
    click.echo("=" * 60)

    all_set = True
    for entry in env_entries:
        status = entry["status"]
        if status == "set":
            if show_all:
                icon = click.style("✓", fg="green")
                click.echo(f"  {icon}  {entry['provider']:<20} ${entry['env_var']} is set")
        elif status == "missing":
            all_set = False
            icon = click.style("✗", fg="red", bold=True)
            click.echo(f"  {icon}  {entry['provider']:<20} ${entry['env_var']} is NOT set")
        elif status == "missing-config":
            all_set = False
            icon = click.style("✗", fg="red", bold=True)
            click.echo(f"  {icon}  {entry['provider']:<20} api_key_env_var is not configured")
        elif status == "no-auth":
            if show_all:
                icon = click.style("-", fg="cyan")
                click.echo(f"  {icon}  {entry['provider']:<20} (no authentication required)")
        elif status == "disabled":
            if show_all:
                icon = click.style("-", fg="white", dim=True)
                click.echo(f"  {icon}  {entry['provider']:<20} (provider disabled)")

    click.echo()
    if all_set:
        click.secho("  ✓ All required environment variables are set.", fg="green")
    else:
        missing = [e for e in env_entries if e["status"] in ("missing", "missing-config")]
        click.secho(
            f"  ✗ {len(missing)} environment variable(s) are missing or unconfigured.",
            fg="red",
        )

    raise SystemExit(0 if all_set else 1)


# ---------------------------------------------------------------------------
# Provider commands
# ---------------------------------------------------------------------------


@main.group()
def provider():
    """Manage model providers.

    List, add, remove, and configure model providers.
    """
    pass


@provider.command("list")
@click.option(
    "--config-file",
    "-c",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Path to configuration file.",
)
@click.option(
    "--builtins",
    is_flag=True,
    default=False,
    help="Show built-in provider templates instead of configured providers.",
)
def provider_list(config_file: Optional[str], builtins: bool):
    """List configured or built-in providers.

    Example:

        claude-code-model-gateway provider list

        claude-code-model-gateway provider list --builtins
    """
    if builtins:
        from src.providers import get_builtin_providers

        providers = get_builtin_providers()
        click.echo("Built-in provider templates:")
        for name, p in sorted(providers.items()):
            model_count = len(p.models)
            click.echo(f"  {p.display_name} ({name}) - {model_count} models")
        return

    from src.config import ConfigError, load_config

    path = Path(config_file) if config_file else None

    try:
        config_obj = load_config(path=path, validate=False)
    except ConfigError as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)

    if not config_obj.providers:
        click.echo("No providers configured.")
        click.echo("Run 'config init' to create a configuration with default providers.")
        return

    click.echo("Configured providers:")
    for name, p in sorted(config_obj.providers.items()):
        status = "enabled" if p.enabled else "disabled"
        default_marker = " (default)" if name == config_obj.default_provider else ""
        model_count = len(p.models)
        click.echo(f"  {p.display_name} ({name}) [{status}]{default_marker} - {model_count} models")


@provider.command("show")
@click.argument("name")
@click.option(
    "--config-file",
    "-c",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Path to configuration file.",
)
def provider_show(name: str, config_file: Optional[str]):
    """Show detailed configuration for a specific provider.

    NAME is the provider identifier (e.g., 'openai', 'anthropic').

    Example:

        claude-code-model-gateway provider show openai
    """
    from src.config import ConfigError, load_config

    path = Path(config_file) if config_file else None

    try:
        config_obj = load_config(path=path, validate=False)
    except ConfigError as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)

    p = config_obj.get_provider(name)
    if p is None:
        click.echo(f"Error: Provider '{name}' not found.", err=True)
        click.echo(
            f"Available providers: {', '.join(config_obj.list_providers())}",
            err=True,
        )
        raise SystemExit(1)

    is_default = name == config_obj.default_provider
    click.echo(f"Provider: {p.display_name} ({p.name})")
    click.echo("=" * 50)
    if is_default:
        click.echo("  ★ Default provider")
    click.echo(f"  Status:        {'enabled' if p.enabled else 'disabled'}")
    click.echo(f"  API base:      {p.api_base}")
    click.echo(f"  Auth type:     {p.auth_type.value}")
    if p.api_key_env_var:
        click.echo(f"  API key env:   ${p.api_key_env_var}")
    click.echo(f"  Default model: {p.default_model or '(none)'}")
    if p.headers:
        click.echo("  Headers:")
        for k, v in p.headers.items():
            click.echo(f"    {k}: {v}")
    if p.extra:
        click.echo("  Extra config:")
        for k, v in p.extra.items():
            click.echo(f"    {k}: {v}")
    if p.models:
        click.echo(f"  Models ({len(p.models)}):")
        for model_name, model in sorted(p.models.items()):
            features = []
            if model.supports_streaming:
                features.append("stream")
            if model.supports_tools:
                features.append("tools")
            if model.supports_vision:
                features.append("vision")
            feature_str = ", ".join(features) if features else "basic"
            click.echo(
                f"    - {model.display_name} ({model_name})"
            )
            click.echo(
                f"      max_tokens={model.max_tokens}, features=[{feature_str}]"
            )


@provider.command("add")
@click.argument("name")
@click.option(
    "--api-base",
    required=True,
    help="Base URL for the provider API.",
)
@click.option(
    "--api-key-env",
    default="",
    help="Environment variable name containing the API key.",
)
@click.option(
    "--default-model",
    default="",
    help="Default model to use.",
)
@click.option(
    "--display-name",
    default="",
    help="Human-readable provider name.",
)
@click.option(
    "--from-builtin",
    type=click.Choice(
        ["openai", "anthropic", "azure", "google", "gemini", "bedrock", "openrouter", "local"]
    ),
    default=None,
    help="Initialize from a built-in provider template.",
)
@click.option(
    "--config-file",
    "-c",
    type=click.Path(dir_okay=False),
    default=None,
    help="Path to configuration file.",
)
def provider_add(
    name: str,
    api_base: str,
    api_key_env: str,
    default_model: str,
    display_name: str,
    from_builtin: Optional[str],
    config_file: Optional[str],
):
    """Add a new provider to the configuration.

    NAME is the unique identifier for this provider.

    Example:

        claude-code-model-gateway provider add my-openai --api-base https://api.openai.com/v1 --api-key-env OPENAI_API_KEY

        claude-code-model-gateway provider add local-llm --api-base http://localhost:8000/v1 --default-model llama3
    """
    from src.config import (
        ConfigError,
        find_config_file,
        load_config,
        save_config_file,
    )
    from src.providers import create_custom_provider, get_builtin_provider

    path = Path(config_file) if config_file else find_config_file()
    if path is None:
        click.echo(
            "Error: No configuration file found. Run 'config init' first.",
            err=True,
        )
        raise SystemExit(1)

    try:
        config_obj = load_config(path=path, validate=False)
    except ConfigError as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)

    if name in config_obj.providers:
        click.echo(f"Error: Provider '{name}' already exists.", err=True)
        click.echo("Remove it first or choose a different name.", err=True)
        raise SystemExit(1)

    if from_builtin:
        new_provider = get_builtin_provider(from_builtin)
        if new_provider is None:
            click.echo(f"Error: Unknown built-in provider '{from_builtin}'.", err=True)
            raise SystemExit(1)
        new_provider.name = name
        if api_base:
            new_provider.api_base = api_base
        if api_key_env:
            new_provider.api_key_env_var = api_key_env
        if default_model:
            new_provider.default_model = default_model
        if display_name:
            new_provider.display_name = display_name
    else:
        new_provider = create_custom_provider(
            name=name,
            api_base=api_base,
            api_key_env_var=api_key_env,
            default_model=default_model,
            display_name=display_name,
        )

    config_obj.add_provider(new_provider)

    try:
        save_config_file(config_obj, path)
        click.echo(f"Provider '{name}' added successfully.")
    except ConfigError as e:
        click.echo(f"Error saving config: {e}", err=True)
        raise SystemExit(1)


@provider.command("remove")
@click.argument("name")
@click.option(
    "--config-file",
    "-c",
    type=click.Path(dir_okay=False),
    default=None,
    help="Path to configuration file.",
)
def provider_remove(name: str, config_file: Optional[str]):
    """Remove a provider from the configuration.

    NAME is the provider identifier to remove.

    Example:

        claude-code-model-gateway provider remove my-openai
    """
    from src.config import (
        ConfigError,
        find_config_file,
        load_config,
        save_config_file,
    )

    path = Path(config_file) if config_file else find_config_file()
    if path is None:
        click.echo(
            "Error: No configuration file found. Run 'config init' first.",
            err=True,
        )
        raise SystemExit(1)

    try:
        config_obj = load_config(path=path, validate=False)
    except ConfigError as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)

    if not config_obj.remove_provider(name):
        click.echo(f"Error: Provider '{name}' not found.", err=True)
        raise SystemExit(1)

    try:
        save_config_file(config_obj, path)
        click.echo(f"Provider '{name}' removed.")
    except ConfigError as e:
        click.echo(f"Error saving config: {e}", err=True)
        raise SystemExit(1)


@provider.command("set-default")
@click.argument("name")
@click.option(
    "--config-file",
    "-c",
    type=click.Path(dir_okay=False),
    default=None,
    help="Path to configuration file.",
)
def provider_set_default(name: str, config_file: Optional[str]):
    """Set the default provider.

    NAME is the provider identifier to make the default.

    Example:

        claude-code-model-gateway provider set-default anthropic
    """
    from src.config import (
        ConfigError,
        find_config_file,
        load_config,
        save_config_file,
    )

    path = Path(config_file) if config_file else find_config_file()
    if path is None:
        click.echo(
            "Error: No configuration file found. Run 'config init' first.",
            err=True,
        )
        raise SystemExit(1)

    try:
        config_obj = load_config(path=path, validate=False)
    except ConfigError as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)

    if name not in config_obj.providers:
        click.echo(f"Error: Provider '{name}' not found.", err=True)
        click.echo(
            f"Available providers: {', '.join(config_obj.list_providers())}",
            err=True,
        )
        raise SystemExit(1)

    config_obj.default_provider = name

    try:
        save_config_file(config_obj, path)
        click.echo(f"Default provider set to '{name}'.")
    except ConfigError as e:
        click.echo(f"Error saving config: {e}", err=True)
        raise SystemExit(1)


@provider.command("enable")
@click.argument("name")
@click.option(
    "--config-file",
    "-c",
    type=click.Path(dir_okay=False),
    default=None,
    help="Path to configuration file.",
)
def provider_enable(name: str, config_file: Optional[str]):
    """Enable a provider.

    Example:

        claude-code-model-gateway provider enable openai
    """
    _set_provider_enabled(name, True, config_file)


@provider.command("disable")
@click.argument("name")
@click.option(
    "--config-file",
    "-c",
    type=click.Path(dir_okay=False),
    default=None,
    help="Path to configuration file.",
)
def provider_disable(name: str, config_file: Optional[str]):
    """Disable a provider.

    Example:

        claude-code-model-gateway provider disable openai
    """
    _set_provider_enabled(name, False, config_file)


def _set_provider_enabled(
    name: str, enabled: bool, config_file: Optional[str]
) -> None:
    """Enable or disable a provider."""
    from src.config import (
        ConfigError,
        find_config_file,
        load_config,
        save_config_file,
    )

    path = Path(config_file) if config_file else find_config_file()
    if path is None:
        click.echo(
            "Error: No configuration file found. Run 'config init' first.",
            err=True,
        )
        raise SystemExit(1)

    try:
        config_obj = load_config(path=path, validate=False)
    except ConfigError as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)

    p = config_obj.get_provider(name)
    if p is None:
        click.echo(f"Error: Provider '{name}' not found.", err=True)
        raise SystemExit(1)

    p.enabled = enabled
    action = "enabled" if enabled else "disabled"

    try:
        save_config_file(config_obj, path)
        click.echo(f"Provider '{name}' {action}.")
    except ConfigError as e:
        click.echo(f"Error saving config: {e}", err=True)
        raise SystemExit(1)


@provider.command("update")
@click.argument("name")
@click.option(
    "--api-base",
    default=None,
    help="New base URL for the provider API.",
)
@click.option(
    "--api-key-env",
    default=None,
    help="New environment variable name containing the API key.",
)
@click.option(
    "--display-name",
    default=None,
    help="New human-readable provider name.",
)
@click.option(
    "--default-model",
    default=None,
    help="New default model to use.",
)
@click.option(
    "--auth-type",
    type=click.Choice(["api_key", "bearer_token", "none"]),
    default=None,
    help="New authentication type.",
)
@click.option(
    "--config-file",
    "-c",
    type=click.Path(dir_okay=False),
    default=None,
    help="Path to configuration file.",
)
def provider_update(
    name: str,
    api_base: Optional[str],
    api_key_env: Optional[str],
    display_name: Optional[str],
    default_model: Optional[str],
    auth_type: Optional[str],
    config_file: Optional[str],
):
    """Update properties of an existing provider.

    NAME is the provider identifier to update.

    Only options that are explicitly provided are changed; others remain
    unchanged.

    Example:

        claude-code-model-gateway provider update openai --api-base https://my-proxy.example.com/v1

        claude-code-model-gateway provider update local-llm --default-model llama3 --display-name "Local LLaMA"
    """
    from src.config import (
        ConfigError,
        find_config_file,
        load_config,
        save_config_file,
    )
    from src.config.manager import ConfigManagerError, ConfigManager, ProviderNotFoundError
    from src.models import AuthType

    path = Path(config_file) if config_file else find_config_file()
    if path is None:
        click.echo(
            "Error: No configuration file found. Run 'config init' first.",
            err=True,
        )
        raise SystemExit(1)

    try:
        config_obj = load_config(path=path, validate=False)
    except ConfigError as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)

    if name not in config_obj.providers:
        click.echo(f"Error: Provider '{name}' not found.", err=True)
        click.echo(
            f"Available providers: {', '.join(sorted(config_obj.providers.keys()))}",
            err=True,
        )
        raise SystemExit(1)

    provider = config_obj.providers[name]
    changed: list[str] = []

    if api_base is not None:
        provider.api_base = api_base
        changed.append(f"api_base = {api_base}")
    if api_key_env is not None:
        provider.api_key_env_var = api_key_env
        changed.append(f"api_key_env_var = {api_key_env}")
    if display_name is not None:
        provider.display_name = display_name
        changed.append(f"display_name = {display_name}")
    if default_model is not None:
        if default_model and provider.models and default_model not in provider.models:
            click.echo(
                f"Warning: model '{default_model}' is not in the provider's model list.",
                err=True,
            )
        provider.default_model = default_model
        changed.append(f"default_model = {default_model}")
    if auth_type is not None:
        try:
            provider.auth_type = AuthType(auth_type)
        except ValueError:
            click.echo(f"Error: Invalid auth_type '{auth_type}'.", err=True)
            raise SystemExit(1)
        changed.append(f"auth_type = {auth_type}")

    if not changed:
        click.echo("No changes made. Provide at least one option to update.")
        raise SystemExit(0)

    try:
        save_config_file(config_obj, path)
        click.echo(f"Provider '{name}' updated:")
        for change in changed:
            click.echo(f"  {change}")
    except ConfigError as e:
        click.echo(f"Error saving config: {e}", err=True)
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# Provider model sub-commands
# ---------------------------------------------------------------------------


@provider.group("model")
def provider_model():
    """Manage models within a provider.

    Add, remove, list, and update model configurations for a specific provider.
    """
    pass


@provider_model.command("list")
@click.argument("provider_name")
@click.option(
    "--config-file",
    "-c",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Path to configuration file.",
)
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["text", "json"]),
    default="text",
    show_default=True,
    help="Output format.",
)
def provider_model_list(provider_name: str, config_file: Optional[str], fmt: str):
    """List models configured for PROVIDER_NAME.

    Example:

        claude-code-model-gateway provider model list openai

        claude-code-model-gateway provider model list anthropic --format json
    """
    import json as json_mod

    from src.config import ConfigError, load_config
    from src.config.manager import ConfigManager, ProviderNotFoundError

    path = Path(config_file) if config_file else None

    try:
        config_obj = load_config(path=path, validate=False)
    except ConfigError as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)

    if provider_name not in config_obj.providers:
        click.echo(f"Error: Provider '{provider_name}' not found.", err=True)
        raise SystemExit(1)

    provider = config_obj.providers[provider_name]

    if fmt == "json":
        data = [
            model.to_dict()
            for name, model in sorted(provider.models.items())
        ]
        click.echo(json_mod.dumps({"provider": provider_name, "models": data}, indent=2))
        return

    if not provider.models:
        click.echo(f"No models configured for provider '{provider_name}'.")
        return

    click.echo(f"Models for provider '{provider_name}' ({len(provider.models)}):")
    click.echo("-" * 50)
    for model_name, model in sorted(provider.models.items()):
        default_marker = " (default)" if model_name == provider.default_model else ""
        features = []
        if model.supports_streaming:
            features.append("streaming")
        if model.supports_tools:
            features.append("tools")
        if model.supports_vision:
            features.append("vision")
        feature_str = ", ".join(features) if features else "basic"
        click.echo(f"  {model.display_name} ({model_name}){default_marker}")
        click.echo(f"    max_tokens={model.max_tokens}, features=[{feature_str}]")


@provider_model.command("add")
@click.argument("provider_name")
@click.argument("model_name")
@click.option(
    "--display-name",
    default="",
    help="Human-readable model name.",
)
@click.option(
    "--max-tokens",
    default=4096,
    show_default=True,
    type=int,
    help="Maximum output token limit.",
)
@click.option(
    "--streaming/--no-streaming",
    default=True,
    show_default=True,
    help="Whether the model supports SSE streaming.",
)
@click.option(
    "--tools/--no-tools",
    default=False,
    show_default=True,
    help="Whether the model supports function/tool calling.",
)
@click.option(
    "--vision/--no-vision",
    default=False,
    show_default=True,
    help="Whether the model supports image inputs.",
)
@click.option(
    "--set-default",
    "set_as_default",
    is_flag=True,
    default=False,
    help="Set this model as the provider's default model.",
)
@click.option(
    "--config-file",
    "-c",
    type=click.Path(dir_okay=False),
    default=None,
    help="Path to configuration file.",
)
def provider_model_add(
    provider_name: str,
    model_name: str,
    display_name: str,
    max_tokens: int,
    streaming: bool,
    tools: bool,
    vision: bool,
    set_as_default: bool,
    config_file: Optional[str],
):
    """Add a model to PROVIDER_NAME.

    MODEL_NAME is the unique identifier for the model within the provider.

    Example:

        claude-code-model-gateway provider model add openai gpt-4-turbo --max-tokens 4096 --tools --vision

        claude-code-model-gateway provider model add local-llm llama3 --display-name "LLaMA 3" --set-default
    """
    from src.config import ConfigError, find_config_file, load_config, save_config_file
    from src.config.manager import ModelExistsError, ProviderNotFoundError

    path = Path(config_file) if config_file else find_config_file()
    if path is None:
        click.echo(
            "Error: No configuration file found. Run 'config init' first.",
            err=True,
        )
        raise SystemExit(1)

    try:
        config_obj = load_config(path=path, validate=False)
    except ConfigError as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)

    if provider_name not in config_obj.providers:
        click.echo(f"Error: Provider '{provider_name}' not found.", err=True)
        click.echo(
            f"Available providers: {', '.join(sorted(config_obj.providers.keys()))}",
            err=True,
        )
        raise SystemExit(1)

    provider = config_obj.providers[provider_name]

    if model_name in provider.models:
        click.echo(
            f"Error: Model '{model_name}' already exists on provider '{provider_name}'.",
            err=True,
        )
        click.echo("Remove it first or choose a different name.", err=True)
        raise SystemExit(1)

    from src.models import ModelConfig

    model = ModelConfig(
        name=model_name,
        display_name=display_name or model_name,
        max_tokens=max_tokens,
        supports_streaming=streaming,
        supports_tools=tools,
        supports_vision=vision,
    )
    provider.models[model_name] = model

    # Auto-set as default if provider has no default model, or explicitly requested.
    if set_as_default or not provider.default_model:
        provider.default_model = model_name

    try:
        save_config_file(config_obj, path)
        click.echo(f"Model '{model_name}' added to provider '{provider_name}'.")
        if provider.default_model == model_name:
            click.echo(f"  Default model set to '{model_name}'.")
    except ConfigError as e:
        click.echo(f"Error saving config: {e}", err=True)
        raise SystemExit(1)


@provider_model.command("remove")
@click.argument("provider_name")
@click.argument("model_name")
@click.option(
    "--config-file",
    "-c",
    type=click.Path(dir_okay=False),
    default=None,
    help="Path to configuration file.",
)
def provider_model_remove(provider_name: str, model_name: str, config_file: Optional[str]):
    """Remove a model from PROVIDER_NAME.

    MODEL_NAME is the model identifier to remove.

    If the model is the provider's default, the default is cleared and the
    first remaining model (alphabetically) is promoted automatically.

    Example:

        claude-code-model-gateway provider model remove openai gpt-4-turbo
    """
    from src.config import ConfigError, find_config_file, load_config, save_config_file

    path = Path(config_file) if config_file else find_config_file()
    if path is None:
        click.echo(
            "Error: No configuration file found. Run 'config init' first.",
            err=True,
        )
        raise SystemExit(1)

    try:
        config_obj = load_config(path=path, validate=False)
    except ConfigError as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)

    if provider_name not in config_obj.providers:
        click.echo(f"Error: Provider '{provider_name}' not found.", err=True)
        raise SystemExit(1)

    provider = config_obj.providers[provider_name]

    if model_name not in provider.models:
        click.echo(
            f"Error: Model '{model_name}' not found on provider '{provider_name}'.",
            err=True,
        )
        click.echo(
            f"Available models: {', '.join(sorted(provider.models.keys()))}",
            err=True,
        )
        raise SystemExit(1)

    del provider.models[model_name]

    # Update default_model if it was the removed model.
    if provider.default_model == model_name:
        remaining = sorted(provider.models.keys())
        provider.default_model = remaining[0] if remaining else ""
        if provider.default_model:
            click.echo(f"  Default model updated to '{provider.default_model}'.")
        else:
            click.echo("  No default model set (no models remaining).")

    try:
        save_config_file(config_obj, path)
        click.echo(f"Model '{model_name}' removed from provider '{provider_name}'.")
    except ConfigError as e:
        click.echo(f"Error saving config: {e}", err=True)
        raise SystemExit(1)


@provider_model.command("set-default")
@click.argument("provider_name")
@click.argument("model_name")
@click.option(
    "--config-file",
    "-c",
    type=click.Path(dir_okay=False),
    default=None,
    help="Path to configuration file.",
)
def provider_model_set_default(
    provider_name: str, model_name: str, config_file: Optional[str]
):
    """Set the default model for PROVIDER_NAME.

    MODEL_NAME is the model identifier to make the default.

    Example:

        claude-code-model-gateway provider model set-default openai gpt-4o
    """
    from src.config import ConfigError, find_config_file, load_config, save_config_file

    path = Path(config_file) if config_file else find_config_file()
    if path is None:
        click.echo(
            "Error: No configuration file found. Run 'config init' first.",
            err=True,
        )
        raise SystemExit(1)

    try:
        config_obj = load_config(path=path, validate=False)
    except ConfigError as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)

    if provider_name not in config_obj.providers:
        click.echo(f"Error: Provider '{provider_name}' not found.", err=True)
        raise SystemExit(1)

    provider = config_obj.providers[provider_name]

    if model_name not in provider.models:
        click.echo(
            f"Error: Model '{model_name}' not found on provider '{provider_name}'.",
            err=True,
        )
        click.echo(
            f"Available models: {', '.join(sorted(provider.models.keys()))}",
            err=True,
        )
        raise SystemExit(1)

    provider.default_model = model_name

    try:
        save_config_file(config_obj, path)
        click.echo(
            f"Default model for provider '{provider_name}' set to '{model_name}'."
        )
    except ConfigError as e:
        click.echo(f"Error saving config: {e}", err=True)
        raise SystemExit(1)


@provider_model.command("update")
@click.argument("provider_name")
@click.argument("model_name")
@click.option(
    "--display-name",
    default=None,
    help="New human-readable model name.",
)
@click.option(
    "--max-tokens",
    default=None,
    type=int,
    help="New maximum output token limit.",
)
@click.option(
    "--streaming",
    type=click.BOOL,
    default=None,
    help="Set streaming capability (true/false).",
)
@click.option(
    "--tools",
    type=click.BOOL,
    default=None,
    help="Set tool-calling capability (true/false).",
)
@click.option(
    "--vision",
    type=click.BOOL,
    default=None,
    help="Set vision/image-input capability (true/false).",
)
@click.option(
    "--config-file",
    "-c",
    type=click.Path(dir_okay=False),
    default=None,
    help="Path to configuration file.",
)
def provider_model_update(
    provider_name: str,
    model_name: str,
    display_name: Optional[str],
    max_tokens: Optional[int],
    streaming: Optional[bool],
    tools: Optional[bool],
    vision: Optional[bool],
    config_file: Optional[str],
):
    """Update properties of a model on PROVIDER_NAME.

    MODEL_NAME is the model identifier to update.  Only options that are
    explicitly provided are changed; others remain unchanged.

    Boolean capabilities use true/false values:

    \b
        --streaming true    Enable streaming
        --streaming false   Disable streaming
        --tools true        Enable tool calling
        --vision false      Disable vision inputs

    Example:

        claude-code-model-gateway provider model update openai gpt-4o --max-tokens 8192

        claude-code-model-gateway provider model update openai gpt-4o --streaming false --tools true
    """
    from src.config import ConfigError, find_config_file, load_config, save_config_file

    path = Path(config_file) if config_file else find_config_file()
    if path is None:
        click.echo(
            "Error: No configuration file found. Run 'config init' first.",
            err=True,
        )
        raise SystemExit(1)

    try:
        config_obj = load_config(path=path, validate=False)
    except ConfigError as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)

    if provider_name not in config_obj.providers:
        click.echo(f"Error: Provider '{provider_name}' not found.", err=True)
        raise SystemExit(1)

    provider = config_obj.providers[provider_name]

    if model_name not in provider.models:
        click.echo(
            f"Error: Model '{model_name}' not found on provider '{provider_name}'.",
            err=True,
        )
        click.echo(
            f"Available models: {', '.join(sorted(provider.models.keys()))}",
            err=True,
        )
        raise SystemExit(1)

    model = provider.models[model_name]
    changed: list[str] = []

    if display_name is not None:
        model.display_name = display_name
        changed.append(f"display_name = {display_name}")
    if max_tokens is not None:
        model.max_tokens = max_tokens
        changed.append(f"max_tokens = {max_tokens}")
    if streaming is not None:
        model.supports_streaming = streaming
        changed.append(f"supports_streaming = {streaming}")
    if tools is not None:
        model.supports_tools = tools
        changed.append(f"supports_tools = {tools}")
    if vision is not None:
        model.supports_vision = vision
        changed.append(f"supports_vision = {vision}")

    if not changed:
        click.echo("No changes made. Provide at least one option to update.")
        raise SystemExit(0)

    try:
        save_config_file(config_obj, path)
        click.echo(
            f"Model '{model_name}' on provider '{provider_name}' updated:"
        )
        for change in changed:
            click.echo(f"  {change}")
    except ConfigError as e:
        click.echo(f"Error saving config: {e}", err=True)
        raise SystemExit(1)


@provider_model.command("show")
@click.argument("provider_name")
@click.argument("model_name")
@click.option(
    "--config-file",
    "-c",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Path to configuration file.",
)
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["text", "json"]),
    default="text",
    show_default=True,
    help="Output format.",
)
def provider_model_show(
    provider_name: str,
    model_name: str,
    config_file: Optional[str],
    fmt: str,
):
    """Show detailed configuration for a specific model.

    PROVIDER_NAME and MODEL_NAME identify the model to inspect.

    Example:

        claude-code-model-gateway provider model show openai gpt-4o

        claude-code-model-gateway provider model show anthropic claude-sonnet-4-20250514 --format json
    """
    import json as json_mod

    from src.config import ConfigError, load_config

    path = Path(config_file) if config_file else None

    try:
        config_obj = load_config(path=path, validate=False)
    except ConfigError as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)

    if provider_name not in config_obj.providers:
        click.echo(f"Error: Provider '{provider_name}' not found.", err=True)
        raise SystemExit(1)

    provider = config_obj.providers[provider_name]

    if model_name not in provider.models:
        click.echo(
            f"Error: Model '{model_name}' not found on provider '{provider_name}'.",
            err=True,
        )
        click.echo(
            f"Available models: {', '.join(sorted(provider.models.keys()))}",
            err=True,
        )
        raise SystemExit(1)

    model = provider.models[model_name]

    if fmt == "json":
        data = model.to_dict()
        data["provider"] = provider_name
        data["is_default"] = model_name == provider.default_model
        click.echo(json_mod.dumps(data, indent=2))
        return

    is_default = model_name == provider.default_model
    click.echo(f"Model: {model.display_name} ({model_name})")
    click.echo("=" * 50)
    click.echo(f"  Provider:          {provider_name}")
    if is_default:
        click.echo("  ★ Default model for this provider")
    click.echo(f"  Max tokens:        {model.max_tokens}")
    click.echo(f"  Supports streaming:{' yes' if model.supports_streaming else ' no'}")
    click.echo(f"  Supports tools:    {' yes' if model.supports_tools else ' no'}")
    click.echo(f"  Supports vision:   {' yes' if model.supports_vision else ' no'}")
    if model.extra:
        click.echo("  Extra config:")
        for k, v in model.extra.items():
            click.echo(f"    {k}: {v}")


# ---------------------------------------------------------------------------
# Configuration validation and testing commands
# ---------------------------------------------------------------------------


@main.command("validate-config")
@click.option(
    "--config-file",
    "-c",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Path to configuration file. Auto-discovered if not specified.",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["text", "json"]),
    default="text",
    show_default=True,
    help="Output format for validation results.",
)
@click.option(
    "--strict",
    is_flag=True,
    default=False,
    help="Treat warnings as errors (exit code 1 on warnings).",
)
@click.option(
    "--show-info",
    is_flag=True,
    default=False,
    help="Include informational messages in output.",
)
def validate_config_cmd(
    config_file: Optional[str],
    output_format: str,
    strict: bool,
    show_info: bool,
):
    """Run comprehensive configuration validation.

    Performs detailed validation with error, warning, and informational
    messages. Checks types, value ranges, cross-references, and
    best-practice recommendations.

    Returns exit code 0 on success, 1 on validation errors.

    Example:

        claude-code-model-gateway validate-config

        claude-code-model-gateway validate-config -c gateway.yaml --strict

        claude-code-model-gateway validate-config --format json --show-info
    """
    import json as json_mod
    import sys

    from src.config import ConfigError, load_config
    from src.validation.validator import ConfigValidator, Severity

    path = Path(config_file) if config_file else None

    try:
        config_obj = load_config(path=path, validate=False)
    except ConfigError as e:
        if output_format == "json":
            click.echo(
                json_mod.dumps(
                    {"valid": False, "error": str(e), "messages": []},
                    indent=2,
                )
            )
        else:
            click.secho(f"Error: {e}", fg="red", err=True)
        raise SystemExit(1)

    result = ConfigValidator.validate(config_obj)

    # In strict mode, warnings also cause failure
    has_issues = (
        result.error_count > 0
        or (strict and result.warning_count > 0)
    )

    if output_format == "json":
        output = {
            "valid": result.is_valid and (not strict or result.warning_count == 0),
            "error_count": result.error_count,
            "warning_count": result.warning_count,
            "info_count": len(result.infos),
            "messages": [
                msg.to_dict()
                for msg in result.messages
                if show_info or msg.severity != Severity.INFO
            ],
        }
        click.echo(json_mod.dumps(output, indent=2))
    else:
        click.echo(result.format_report(show_info=show_info))

        if result.is_valid and not has_issues:
            click.secho("\n✓ Configuration is valid.", fg="green")
        elif result.is_valid and has_issues:
            click.secho(
                "\n⚠ Configuration has warnings (strict mode fails).",
                fg="yellow",
            )
        else:
            click.secho("\n✗ Configuration has errors.", fg="red")

    raise SystemExit(1 if has_issues else 0)


@main.command("test-config")
@click.option(
    "--config-file",
    "-c",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Validate a specific config file (in addition to built-in tests).",
)
@click.option(
    "--builtin/--no-builtin",
    default=True,
    help="Run the built-in test suite.",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["text", "json"]),
    default="text",
    show_default=True,
    help="Output format for test results.",
)
def test_config_cmd(
    config_file: Optional[str],
    builtin: bool,
    output_format: str,
):
    """Run configuration validation test suite.

    Runs a comprehensive set of test cases against the configuration
    validator to verify it correctly accepts valid configs and rejects
    invalid ones.

    Optionally validates a specific config file as well.

    Example:

        claude-code-model-gateway test-config

        claude-code-model-gateway test-config -c gateway.yaml

        claude-code-model-gateway test-config --format json

        claude-code-model-gateway test-config --no-builtin -c gateway.yaml
    """
    import json as json_mod
    import sys

    from src.validation.testing import (
        ConfigTestRunner,
        get_builtin_test_cases,
    )

    results = []

    if config_file:
        file_result = ConfigTestRunner.run_file_tests(config_file)
        results.append(file_result)

    if builtin:
        test_cases = get_builtin_test_cases()
        suite_result = ConfigTestRunner.run_suite(test_cases)
        results.extend(suite_result.results)

    if not results:
        click.echo("No tests to run. Provide a config file or use --builtin.")
        raise SystemExit(0)

    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)

    if output_format == "json":
        output = {
            "total": len(results),
            "passed": passed,
            "failed": failed,
            "results": [
                {
                    "name": r.test_case.name,
                    "description": r.test_case.description,
                    "passed": r.passed,
                    "failure_reason": r.failure_reason or None,
                    "error_count": r.validation_result.error_count,
                    "warning_count": r.validation_result.warning_count,
                }
                for r in results
            ],
        }
        click.echo(json_mod.dumps(output, indent=2))
    else:
        click.echo(
            f"Configuration Test Results: {passed}/{len(results)} passed"
        )
        click.echo("=" * 60)

        for r in results:
            status = "PASS" if r.passed else "FAIL"
            color = "green" if r.passed else "red"
            click.secho(f"  [{status}] {r.test_case.name}", fg=color)
            if not r.passed:
                click.echo(f"         {r.failure_reason}")

        click.echo("=" * 60)
        if failed == 0:
            click.secho("All tests passed!", fg="green")
        else:
            click.secho(f"{failed} test(s) failed.", fg="red")

    raise SystemExit(1 if failed > 0 else 0)


@main.command("config-diff")
@click.argument("config_a", type=click.Path(exists=True, dir_okay=False))
@click.argument("config_b", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["text", "json"]),
    default="text",
    show_default=True,
    help="Output format for diff results.",
)
def config_diff_cmd(config_a: str, config_b: str, output_format: str):
    """Compare two configuration files and show differences.

    CONFIG_A and CONFIG_B are paths to configuration files to compare.
    Exit code 0 if identical, 2 if differences found.

    Example:

        claude-code-model-gateway config-diff gateway.yaml gateway-prod.yaml

        claude-code-model-gateway config-diff config.json staging.json --format json
    """
    import json as json_mod
    import sys

    from src.config import ConfigError, load_config_file
    from src.validation.testing import ConfigDiff

    try:
        data_a = load_config_file(Path(config_a))
    except ConfigError as e:
        click.secho(f"Error loading {config_a}: {e}", fg="red", err=True)
        raise SystemExit(1)

    try:
        data_b = load_config_file(Path(config_b))
    except ConfigError as e:
        click.secho(f"Error loading {config_b}: {e}", fg="red", err=True)
        raise SystemExit(1)

    diffs = ConfigDiff.diff(data_a, data_b)

    if output_format == "json":
        output = {
            "file_a": config_a,
            "file_b": config_b,
            "identical": len(diffs) == 0,
            "difference_count": len(diffs),
            "differences": diffs,
        }
        click.echo(json_mod.dumps(output, indent=2))
    else:
        report = ConfigDiff.format_diff(
            data_a, data_b, label_a=config_a, label_b=config_b
        )
        click.echo(report)

    raise SystemExit(0 if len(diffs) == 0 else 2)


# ---------------------------------------------------------------------------
# Cache management commands
# ---------------------------------------------------------------------------


@main.group()
def cache():
    """Manage the application cache.

    View statistics, clear cached data, and configure cache behavior
    for improved performance.
    """
    pass


@cache.command("stats")
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["text", "json"]),
    default="text",
    show_default=True,
    help="Output format.",
)
def cache_stats(fmt: str):
    """Show cache statistics for all active caches.

    Displays hit/miss counts, hit rate, evictions, and current size
    for each registered cache.

    Example:

        claude-code-model-gateway cache stats

        claude-code-model-gateway cache stats --format json
    """
    import json as json_mod

    from src.cache import list_caches

    caches = list_caches()

    if fmt == "json":
        output = {
            "caches": {
                name: c.get_stats().to_dict() for name, c in sorted(caches.items())
            },
            "total_caches": len(caches),
        }
        click.echo(json_mod.dumps(output, indent=2))
        return

    if not caches:
        click.echo("No active caches.")
        return

    click.echo("Cache Statistics")
    click.echo("=" * 60)
    for name, c in sorted(caches.items()):
        stats = c.get_stats()
        click.echo(f"\n  {name}")
        click.echo(f"  {'-' * 40}")
        click.echo(f"    Entries:     {stats.current_size} / {stats.max_size}")
        click.echo(f"    Hits:        {stats.hits}")
        click.echo(f"    Misses:      {stats.misses}")
        click.echo(f"    Hit rate:    {stats.hit_rate:.1f}%")
        click.echo(f"    Evictions:   {stats.evictions}")
        click.echo(f"    Expirations: {stats.expirations}")
        click.echo(f"    Sets:        {stats.sets}")
    click.echo()


@cache.command("clear")
@click.argument("name", required=False, default=None)
@click.option(
    "--all",
    "clear_all",
    is_flag=True,
    default=False,
    help="Clear all caches.",
)
def cache_clear(name: Optional[str], clear_all: bool):
    """Clear cached data.

    With no arguments, lists available caches. Provide a cache NAME
    to clear a specific cache, or use --all to clear everything.

    Example:

        claude-code-model-gateway cache clear --all

        claude-code-model-gateway cache clear config

        claude-code-model-gateway cache clear providers
    """
    from src.cache import clear_all_caches, list_caches

    if clear_all:
        results = clear_all_caches()
        if not results:
            click.echo("No active caches to clear.")
        else:
            total = sum(results.values())
            click.echo(f"Cleared {total} entries from {len(results)} cache(s):")
            for cache_name, count in sorted(results.items()):
                click.echo(f"  {cache_name}: {count} entries removed")
        return

    if name is None:
        caches = list_caches()
        if not caches:
            click.echo("No active caches.")
            click.echo("Use --all to clear all caches when they are created.")
        else:
            click.echo("Available caches:")
            for cache_name, c in sorted(caches.items()):
                stats = c.get_stats()
                click.echo(f"  {cache_name} ({stats.current_size} entries)")
            click.echo()
            click.echo("Specify a cache name or use --all to clear.")
        return

    caches = list_caches()
    if name not in caches:
        click.echo(f"Error: Cache '{name}' not found.", err=True)
        if caches:
            click.echo(
                f"Available caches: {', '.join(sorted(caches.keys()))}", err=True
            )
        raise SystemExit(1)

    count = caches[name].clear()
    click.echo(f"Cleared {count} entries from cache '{name}'.")


@cache.command("purge")
def cache_purge():
    """Remove expired entries from all caches.

    Scans all active caches and removes entries that have exceeded
    their time-to-live (TTL), freeing memory without affecting
    unexpired entries.

    Example:

        claude-code-model-gateway cache purge
    """
    from src.cache import list_caches

    caches = list_caches()
    if not caches:
        click.echo("No active caches.")
        return

    total = 0
    for name, c in sorted(caches.items()):
        purged = c.purge_expired()
        if purged > 0:
            click.echo(f"  {name}: purged {purged} expired entries")
        total += purged

    if total == 0:
        click.echo("No expired entries found.")
    else:
        click.echo(f"\nPurged {total} expired entries total.")


@cache.command("warmup")
@click.option(
    "--config",
    "config_path",
    default=None,
    type=click.Path(dir_okay=False),
    help="Path to configuration file to warm into cache.",
)
@click.option(
    "--providers/--no-providers",
    default=True,
    show_default=True,
    help="Warm built-in provider cache.",
)
@click.option(
    "--parallel/--sequential",
    default=False,
    show_default=True,
    help="Warm entries in parallel for faster startup.",
)
def cache_warmup(config_path: Optional[str], providers: bool, parallel: bool):
    """Pre-populate caches for faster first requests.

    Loads frequently-accessed data into the cache at startup time
    so the first requests don't pay the cost of cold cache misses.

    Example:

        claude-code-model-gateway cache warmup

        claude-code-model-gateway cache warmup --config gateway.yaml --parallel
    """
    from src.cache import CacheWarmer, get_config_cache, get_provider_cache

    warmer = CacheWarmer(name="cli_warmup")
    warmed_caches = []

    if providers:
        def _load_providers():
            from src.providers import get_builtin_providers
            return get_builtin_providers(use_cache=False)

        warmer.add("builtin_providers", _load_providers)
        warmed_caches.append("providers")

    if config_path:
        def _load_config():
            from src.config import load_config
            return load_config(path=Path(config_path), validate=False, use_cache=False)

        warmer.add(f"config:{config_path}", _load_config)
        warmed_caches.append("config")

    if len(warmer) == 0:
        click.echo("Nothing to warm. Use --providers or --config to specify targets.")
        return

    click.echo(f"Warming caches: {', '.join(warmed_caches)} ...")

    # Warm providers into the provider cache
    if providers:
        provider_cache = get_provider_cache()
        results = warmer.warmup(provider_cache, parallel=parallel)
    else:
        config_cache = get_config_cache()
        results = warmer.warmup(config_cache, parallel=parallel)

    succeeded = sum(1 for v in results.values() if v)
    failed = sum(1 for v in results.values() if not v)

    click.echo(f"Warmup complete: {succeeded} succeeded, {failed} failed.")
    for key, success in sorted(results.items()):
        status = click.style("OK", fg="green") if success else click.style("FAIL", fg="red")
        click.echo(f"  {key}: {status}")


@cache.command("info")
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["text", "json"]),
    default="text",
    show_default=True,
    help="Output format.",
)
def cache_info(fmt: str):
    """Show detailed cache information including configuration.

    Displays cache names, sizes, TTL settings, and advanced features
    like stale-while-revalidate and compression status.

    Example:

        claude-code-model-gateway cache info

        claude-code-model-gateway cache info --format json
    """
    import json as json_mod

    from src.cache import list_caches

    caches = list_caches()

    if fmt == "json":
        info = {}
        for name, c in sorted(caches.items()):
            stats = c.get_stats()
            info[name] = {
                "maxsize": c.maxsize,
                "default_ttl": c.default_ttl,
                "stale_ttl": c.stale_ttl,
                "current_size": stats.current_size,
                "stats": stats.to_dict(),
            }
        click.echo(json_mod.dumps({"caches": info, "total_caches": len(caches)}, indent=2))
        return

    if not caches:
        click.echo("No active caches.")
        return

    click.echo("Cache Information")
    click.echo("=" * 60)
    for name, c in sorted(caches.items()):
        stats = c.get_stats()
        click.echo(f"\n  {name}")
        click.echo(f"  {'-' * 40}")
        click.echo(f"    Max size:      {c.maxsize}")
        click.echo(f"    Default TTL:   {c.default_ttl}s")
        click.echo(f"    Stale TTL:     {c.stale_ttl}s")
        click.echo(f"    Current size:  {stats.current_size}")
        click.echo(f"    Hit rate:      {stats.hit_rate:.1f}%")
        click.echo(f"    Stale hits:    {stats.stale_hits}")
        if stats.compressed_entries > 0:
            click.echo(f"    Compressed:    {stats.compressed_entries}")
    click.echo()


@cache.command("response-stats")
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["text", "json"]),
    default="text",
    show_default=True,
    help="Output format.",
)
def cache_response_stats(fmt: str):
    """Show HTTP response cache statistics.

    Displays lookup counts, hit rates, bytes saved, and compression
    statistics for the response cache.

    Example:

        claude-code-model-gateway cache response-stats

        claude-code-model-gateway cache response-stats --format json
    """
    import json as json_mod

    from src.response_cache import get_response_cache

    rc = get_response_cache()
    stats = rc.get_stats()

    if fmt == "json":
        click.echo(json_mod.dumps(stats.to_dict(), indent=2))
        return

    click.echo("Response Cache Statistics")
    click.echo("=" * 60)
    click.echo(f"  Lookups:           {stats.lookups}")
    click.echo(f"  Hits:              {stats.hits}")
    click.echo(f"  Misses:            {stats.misses}")
    click.echo(f"  Bypasses:          {stats.bypasses}")
    click.echo(f"  Hit rate:          {stats.hit_rate:.1f}%")
    click.echo(f"  Stores:            {stats.stores}")
    click.echo(f"  Compressed stores: {stats.compressed_stores}")
    click.echo(f"  Bytes saved:       {stats.bytes_saved:,}")
    click.echo()


@cache.command("token-count-stats")
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["text", "json"]),
    default="text",
    show_default=True,
    help="Output format.",
)
def cache_token_count_stats(fmt: str):
    """Show token count cache statistics.

    Displays lookup counts, hit rates, bytes saved, and compression
    statistics for the POST /v1/messages/count_tokens cache.

    Example:

        claude-code-model-gateway cache token-count-stats

        claude-code-model-gateway cache token-count-stats --format json
    """
    import json as json_mod

    from src.token_count_cache import get_token_count_cache

    tc = get_token_count_cache()
    stats = tc.get_stats()

    if fmt == "json":
        click.echo(json_mod.dumps(stats.to_dict(), indent=2))
        return

    click.echo("Token Count Cache Statistics")
    click.echo("=" * 60)
    click.echo(f"  Lookups:           {stats.lookups}")
    click.echo(f"  Hits:              {stats.hits}")
    click.echo(f"  Misses:            {stats.misses}")
    click.echo(f"  Bypasses:          {stats.bypasses}")
    click.echo(f"  Hit rate:          {stats.hit_rate:.1f}%")
    click.echo(f"  Stores:            {stats.stores}")
    click.echo(f"  Compressed stores: {stats.compressed_stores}")
    click.echo(f"  Bytes saved:       {stats.bytes_saved:,}")
    click.echo()


# ---------------------------------------------------------------------------
# Logging management commands
# ---------------------------------------------------------------------------


@main.group("logging")
def logging_group():
    """Manage application logging.

    View logging status, test log output, and configure logging behavior.
    """
    pass


@logging_group.command("status")
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["text", "json"]),
    default="text",
    show_default=True,
    help="Output format.",
)
def logging_status(fmt: str):
    """Show the current logging system status.

    Displays configured log level, active handlers, formatters,
    and filters.

    Example:

        claude-code-model-gateway logging status

        claude-code-model-gateway logging status --format json
    """
    import json as json_mod

    from src.logging_config import get_logging_status, is_configured

    status = get_logging_status()

    if fmt == "json":
        click.echo(json_mod.dumps(status, indent=2))
        return

    click.echo("Logging System Status")
    click.echo("=" * 50)
    click.echo(f"  Configured:      {status['configured']}")
    click.echo(f"  Root logger:     {status['root_logger']}")
    click.echo(f"  Effective level: {status['effective_level']}")
    click.echo(f"  Handler count:   {status['handler_count']}")

    if status["handlers"]:
        click.echo()
        click.echo("Handlers:")
        click.echo("-" * 50)
        for i, handler in enumerate(status["handlers"], 1):
            click.echo(f"  [{i}] {handler['type']}")
            click.echo(f"      Level:     {handler['level']}")
            click.echo(f"      Formatter: {handler['formatter']}")
            if handler.get("file"):
                click.echo(f"      File:      {handler['file']}")
                click.echo(f"      Max size:  {handler['max_bytes']} bytes")
                click.echo(f"      Backups:   {handler['backup_count']}")
            if handler["filters"]:
                click.echo(f"      Filters:   {', '.join(handler['filters'])}")

    if status.get("config"):
        click.echo()
        click.echo("Active Configuration:")
        click.echo("-" * 50)
        config = status["config"]
        click.echo(f"  Level:          {config['level']}")
        click.echo(f"  Format:         {config['log_format']}")
        click.echo(f"  Output:         {config['output']}")
        if config.get("log_file"):
            click.echo(f"  Log file:       {config['log_file']}")
        click.echo(f"  Correlation ID: {config['include_correlation_id']}")
        if config["rate_limit_seconds"] > 0:
            click.echo(f"  Rate limit:     {config['rate_limit_seconds']}s")
        if config["module_filters"]:
            click.echo(f"  Module filters: {', '.join(config['module_filters'])}")


@logging_group.command("test")
@click.option(
    "--level",
    type=click.Choice(["debug", "info", "warning", "error", "critical"]),
    default="info",
    show_default=True,
    help="Log level for setup.",
)
@click.option(
    "--format",
    "log_format",
    type=click.Choice(["standard", "detailed", "json", "colored", "minimal"]),
    default="standard",
    show_default=True,
    help="Log format to test.",
)
def logging_test(level: str, log_format: str):
    """Emit test log messages at every level.

    Useful for verifying that logging is working and to preview
    how different formats and levels look.

    Example:

        claude-code-model-gateway logging test

        claude-code-model-gateway logging test --format json --level debug
    """
    from src.logging_config import (
        correlation_context,
        setup_logging,
    )

    setup_logging(level=level, log_format=log_format, output="console")
    test_logger = get_logger("test")

    click.echo(f"Testing log output (level={level}, format={log_format}):")
    click.echo("-" * 50)

    test_logger.debug("This is a DEBUG message - detailed diagnostic info")
    test_logger.info("This is an INFO message - normal operation")
    test_logger.warning("This is a WARNING message - something unexpected")
    test_logger.error("This is an ERROR message - something went wrong")
    test_logger.critical("This is a CRITICAL message - system failure")

    # Test with correlation ID
    click.echo()
    click.echo("With correlation ID:")
    with correlation_context("test-req-001"):
        test_logger.info("Request processed with correlation ID")

    # Test with extra data
    click.echo()
    click.echo("With extra structured data:")
    test_logger.info(
        "Request completed",
        extra={
            "method": "POST",
            "path": "/v1/messages",
            "status_code": 200,
            "duration_ms": 42.5,
        },
    )

    click.echo("-" * 50)
    click.echo("Log test complete.")


@logging_group.command("formats")
def logging_formats():
    """Show examples of all available log formats.

    Displays a sample log line in each supported format
    (standard, detailed, minimal, json, colored) so you
    can choose the best one for your use case.

    Example:

        claude-code-model-gateway logging formats
    """
    from src.logging_config import reset_logging, setup_logging

    sample_formats = ["standard", "detailed", "minimal", "json", "colored"]

    click.echo("Available Log Formats")
    click.echo("=" * 60)

    for fmt in sample_formats:
        click.echo()
        click.echo(f"  Format: {fmt}")
        click.echo(f"  {'-' * 56}")

        reset_logging()
        setup_logging(level="info", log_format=fmt, output="console")
        demo_logger = get_logger("demo")
        demo_logger.info("Sample log message from the gateway")

        click.echo()

    # Clean up
    reset_logging()
    click.echo("=" * 60)
    click.echo(
        "Use --log-format <name> with proxy or gateway commands "
        "to select a format."
    )


@logging_group.command("levels")
def logging_levels():
    """Show available log levels and their descriptions.

    Example:

        claude-code-model-gateway logging levels
    """
    click.echo("Available Log Levels")
    click.echo("=" * 50)
    levels = [
        ("debug", "10", "Detailed diagnostic information"),
        ("info", "20", "Normal operational messages"),
        ("warning", "30", "Something unexpected happened"),
        ("error", "40", "An error occurred, functionality impacted"),
        ("critical", "50", "System is in a critical state"),
    ]
    for name, num, desc in levels:
        click.echo(f"  {name:<10} ({num})  {desc}")
    click.echo()
    click.echo("Set via: config set log_level <level>")
    click.echo("Override: --verbose flag sets level to debug")


@logging_group.command("metrics")
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["text", "json"]),
    default="text",
    show_default=True,
    help="Output format.",
)
def logging_metrics(fmt: str):
    """Show log metrics and statistics.

    Displays counts of log messages by level, top logging modules,
    error rate, and recent errors.

    Example:

        claude-code-model-gateway logging metrics

        claude-code-model-gateway logging metrics --format json
    """
    import json as json_mod

    from src.logging_config import get_log_metrics

    metrics = get_log_metrics()
    report = metrics.get_report()

    if fmt == "json":
        click.echo(json_mod.dumps(report, indent=2, default=str))
        return

    click.echo("Log Metrics Report")
    click.echo("=" * 50)
    click.echo(f"  Total messages:    {report['total_count']}")
    click.echo(f"  Uptime:            {report['uptime_seconds']}s")
    click.echo(f"  Error rate:        {report['error_rate_per_minute']}/min")

    if report["levels"]:
        click.echo()
        click.echo("Messages by Level:")
        click.echo("-" * 50)
        for level_name, count in sorted(
            report["levels"].items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            click.echo(f"  {level_name:<12} {count}")

    if report["top_modules"]:
        click.echo()
        click.echo("Top Modules:")
        click.echo("-" * 50)
        for module_name, count in list(report["top_modules"].items())[:10]:
            click.echo(f"  {module_name:<30} {count}")

    if report["recent_errors"]:
        click.echo()
        click.echo("Recent Errors:")
        click.echo("-" * 50)
        for error in report["recent_errors"][-5:]:
            click.echo(f"  [{error.get('level', 'ERROR')}] {error.get('module', '?')}: "
                        f"{error.get('message', '?')[:80]}")


@logging_group.command("health")
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["text", "json"]),
    default="text",
    show_default=True,
    help="Output format.",
)
def logging_health(fmt: str):
    """Show logging system health information.

    Displays health status including uptime, message counts,
    error rates, and handler status.

    Example:

        claude-code-model-gateway logging health

        claude-code-model-gateway logging health --format json
    """
    import json as json_mod

    from src.logging_config import (
        get_log_metrics,
        get_logging_status,
    )

    status = get_logging_status()
    metrics = get_log_metrics()
    metrics_report = metrics.get_report()

    health_data = {
        "status": "healthy",
        "logging_configured": status["configured"],
        "effective_level": status["effective_level"],
        "handler_count": status["handler_count"],
        "total_messages": metrics_report["total_count"],
        "error_rate_per_minute": metrics_report["error_rate_per_minute"],
        "levels": metrics_report["levels"],
        "uptime_seconds": metrics_report["uptime_seconds"],
    }

    # Determine overall health
    if metrics_report["error_rate_per_minute"] > 100:
        health_data["status"] = "critical"
    elif metrics_report["error_rate_per_minute"] > 10:
        health_data["status"] = "degraded"

    if fmt == "json":
        click.echo(json_mod.dumps(health_data, indent=2, default=str))
        return

    status_color = {
        "healthy": "green",
        "degraded": "yellow",
        "critical": "red",
    }
    color = status_color.get(health_data["status"], "white")

    click.echo("Logging System Health")
    click.echo("=" * 50)
    click.secho(f"  Status:          {health_data['status']}", fg=color)
    click.echo(f"  Configured:      {health_data['logging_configured']}")
    click.echo(f"  Level:           {health_data['effective_level']}")
    click.echo(f"  Handlers:        {health_data['handler_count']}")
    click.echo(f"  Total messages:  {health_data['total_messages']}")
    click.echo(f"  Error rate:      {health_data['error_rate_per_minute']}/min")
    click.echo(f"  Uptime:          {health_data['uptime_seconds']}s")

    if health_data["levels"]:
        click.echo()
        click.echo("Message Counts:")
        for level_name, count in sorted(health_data["levels"].items()):
            click.echo(f"    {level_name}: {count}")


@logging_group.command("test-redaction")
def logging_test_redaction():
    """Test sensitive data redaction in log output.

    Emits test log messages containing sample sensitive data
    to verify that the SensitiveDataFilter is working correctly.

    Example:

        claude-code-model-gateway logging test-redaction
    """
    from src.logging_config import (
        SensitiveDataFilter,
        reset_logging,
        setup_logging,
    )

    reset_logging()
    setup_logging(level="debug", log_format="standard", output="console")

    # Add sensitive data filter
    root = logging.getLogger("gateway")
    sensitive_filter = SensitiveDataFilter()
    for handler in root.handlers:
        handler.addFilter(sensitive_filter)

    test_logger = get_logger("redaction_test")

    click.echo("Testing Sensitive Data Redaction")
    click.echo("=" * 50)
    click.echo()
    click.echo("The following messages contain sample sensitive data.")
    click.echo("If redaction is working, secrets will be replaced with ***REDACTED***")
    click.echo("-" * 50)

    test_logger.info("API key: sk-ant-abcdef1234567890abcdef1234567890")
    test_logger.info("Setting api_key = my-secret-api-key-value-here")
    test_logger.info("Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.test")
    test_logger.info("password = super_secret_password_123")
    test_logger.info("Normal message without sensitive data")

    click.echo("-" * 50)
    click.echo("Redaction test complete.")

    reset_logging()


@logging_group.command("set-level")
@click.argument(
    "level",
    type=click.Choice(["debug", "info", "warning", "error", "critical"]),
)
def logging_set_level(level: str):
    """Change the active log level at runtime.

    Updates the root logger and all managed handlers to the new level
    without reconfiguring the entire logging system.

    LEVEL must be one of: debug, info, warning, error, critical.

    Example:

        claude-code-model-gateway logging set-level debug

        claude-code-model-gateway logging set-level info
    """
    numeric = set_log_level(level)
    click.echo(f"Log level changed to {level.upper()} ({numeric})")
    status = get_logging_status()
    click.echo(f"  Effective level: {status['effective_level']}")
    click.echo(f"  Configured:      {status['configured']}")


@logging_group.command("configure")
@click.option(
    "--level",
    type=click.Choice(["debug", "info", "warning", "error", "critical"]),
    default=None,
    help="Log level to apply.",
)
@click.option(
    "--log-format",
    type=click.Choice(["standard", "detailed", "json", "colored", "minimal"]),
    default=None,
    help="Log format.",
)
@click.option(
    "--output",
    type=click.Choice(["console", "file", "both"]),
    default=None,
    help="Output destination.",
)
@click.option(
    "--log-file",
    default=None,
    type=click.Path(dir_okay=False),
    help="Path to log file.",
)
@click.option(
    "--rotation-mode",
    type=click.Choice(["size", "time"]),
    default=None,
    help="File rotation mode.",
)
@click.option(
    "--rotation-when",
    default=None,
    help="When to rotate (time mode): midnight, H, D, W0-W6.",
)
@click.option(
    "--from-env",
    is_flag=True,
    default=False,
    help="Load configuration from GATEWAY_LOG_* environment variables.",
)
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["text", "json"]),
    default="text",
    show_default=True,
    help="Output format for the result.",
)
def logging_configure(
    level: Optional[str],
    log_format: Optional[str],
    output: Optional[str],
    log_file: Optional[str],
    rotation_mode: Optional[str],
    rotation_when: Optional[str],
    from_env: bool,
    fmt: str,
):
    """Configure or reconfigure the logging system.

    Applies new logging settings.  Can load settings from environment
    variables (GATEWAY_LOG_*) with --from-env, or accept individual
    options.  Explicitly provided options override environment values.

    Example:

        claude-code-model-gateway logging configure --level debug

        claude-code-model-gateway logging configure --log-format json --output both

        claude-code-model-gateway logging configure --from-env

        GATEWAY_LOG_LEVEL=debug \\
          claude-code-model-gateway logging configure --from-env
    """
    import json as json_mod

    from src.logging_config import reset_logging

    if from_env:
        cfg = LoggingConfig.from_env()
    else:
        cfg = LoggingConfig()

    if level is not None:
        cfg.level = level
    if log_format is not None:
        cfg.log_format = log_format
    if output is not None:
        cfg.output = output
    if log_file is not None:
        cfg.log_file = log_file
    if rotation_mode is not None:
        cfg.rotation_mode = rotation_mode
    if rotation_when is not None:
        cfg.rotation_when = rotation_when

    reset_logging()
    setup_logging(config=cfg)
    status = get_logging_status()

    if fmt == "json":
        click.echo(json_mod.dumps(status, indent=2))
        return

    click.echo("Logging reconfigured:")
    click.echo(f"  Level:    {cfg.level.upper()}")
    click.echo(f"  Format:   {cfg.log_format}")
    click.echo(f"  Output:   {cfg.output}")
    if cfg.log_file:
        click.echo(f"  Log file: {cfg.log_file}")
        click.echo(f"  Rotation: {cfg.rotation_mode}")
    click.echo(f"  Handlers: {status['handler_count']}")


@logging_group.command("files")
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["text", "json"]),
    default="text",
    show_default=True,
    help="Output format.",
)
def logging_files(fmt: str):
    """List active log files and rotation metadata.

    Shows all file-based log handlers with paths, sizes, rotation
    settings, and existing backup files.

    Example:

        claude-code-model-gateway logging files

        claude-code-model-gateway logging files --format json
    """
    import json as json_mod

    files = get_log_files()

    if fmt == "json":
        click.echo(json_mod.dumps(files, indent=2, default=str))
        return

    if not files:
        click.echo("No file-based log handlers are active.")
        click.echo(
            "Use --log-file with proxy/gateway commands, or "
            "'logging configure --output file' to enable file logging."
        )
        return

    click.echo("Active Log Files")
    click.echo("=" * 60)
    for f in files:
        click.echo(f"\n  File:     {f['path']}")
        size_kb = f["size_bytes"] / 1024.0
        click.echo(f"  Size:     {size_kb:.1f} KB ({f['size_bytes']} bytes)")
        click.echo(f"  Rotation: {f['rotation_mode']}")
        if f["rotation_mode"] == "time":
            click.echo(f"  When:     {f.get('rotation_when', 'midnight')}")
        else:
            max_mb = f.get("max_bytes", 0) / (1024 * 1024)
            click.echo(f"  Max size: {max_mb:.0f} MB")
        click.echo(f"  Backups:  {f['backup_count']}")
        if f["backup_files"]:
            click.echo(f"  Backup files ({len(f['backup_files'])}):")
            for bf in f["backup_files"][:5]:
                click.echo(f"    {bf}")
            if len(f["backup_files"]) > 5:
                click.echo(f"    ... and {len(f['backup_files']) - 5} more")


# --------------------------------------------------------------------------- #
# Health & Error Status commands
# --------------------------------------------------------------------------- #


@main.group()
def health():
    """Health monitoring and error tracking commands.

    View system health, provider status, error rates, and
    circuit breaker states.
    """
    pass


@health.command("status")
@click.option(
    "--format", "-f", "fmt",
    type=click.Choice(["text", "json"]),
    default="text",
    show_default=True,
    help="Output format.",
)
def health_status(fmt: str):
    """Show current system and provider health status.

    Displays overall system health, per-provider health metrics,
    error rates, success rates, and circuit breaker states.

    Example:

        claude-code-model-gateway health status
        claude-code-model-gateway health status --format json
    """
    import json as json_mod

    from src.error_handling import get_health_status

    system_health = get_health_status()

    if fmt == "json":
        click.echo(json_mod.dumps(system_health.to_dict(), indent=2))
        return

    # Text format
    state_colors = {
        "healthy": "green",
        "degraded": "yellow",
        "unhealthy": "red",
    }
    state_color = state_colors.get(system_health.state.value, "white")

    click.echo("System Health Status")
    click.echo("=" * 50)
    click.echo(
        f"  State: {click.style(system_health.state.value.upper(), fg=state_color, bold=True)}"
    )
    click.echo(f"  Uptime: {system_health.uptime_seconds:.1f}s")
    click.echo(f"  Total Requests: {system_health.total_requests}")
    click.echo(f"  Total Errors: {system_health.total_errors}")
    click.echo(f"  Error Rate: {system_health.error_rate:.4f}/s")
    click.echo()

    if system_health.providers:
        click.echo("Provider Health")
        click.echo("-" * 50)
        for name, ph in system_health.providers.items():
            p_color = state_colors.get(ph.state.value, "white")
            click.echo(
                f"  {name}: "
                f"{click.style(ph.state.value.upper(), fg=p_color, bold=True)}"
            )
            click.echo(f"    Success Rate: {ph.success_rate:.1%}")
            click.echo(f"    Error Rate: {ph.error_rate:.4f}/s")
            click.echo(f"    Avg Latency: {ph.avg_latency_ms:.1f}ms")
            click.echo(f"    Consecutive Failures: {ph.consecutive_failures}")
            if ph.circuit_state:
                cb_color = (
                    "green" if ph.circuit_state == "closed"
                    else "red" if ph.circuit_state == "open"
                    else "yellow"
                )
                click.echo(
                    f"    Circuit Breaker: "
                    f"{click.style(ph.circuit_state.upper(), fg=cb_color)}"
                )
            if ph.last_error_message:
                click.echo(f"    Last Error: {ph.last_error_message[:80]}")
            click.echo()
    else:
        click.echo("  No provider activity recorded yet.")


@health.command("errors")
@click.option(
    "--provider", "-p",
    default=None,
    help="Filter errors by provider name.",
)
@click.option(
    "--limit", "-n",
    default=20,
    show_default=True,
    type=int,
    help="Maximum number of errors to show.",
)
@click.option(
    "--format", "-f", "fmt",
    type=click.Choice(["text", "json"]),
    default="text",
    show_default=True,
    help="Output format.",
)
def health_errors(provider: Optional[str], limit: int, fmt: str):
    """Show recent error events.

    Lists the most recent errors with details including provider,
    category, severity, and error message.

    Example:

        claude-code-model-gateway health errors
        claude-code-model-gateway health errors --provider anthropic --limit 10
    """
    import json as json_mod

    from src.error_handling import get_error_tracker

    tracker = get_error_tracker()
    errors = tracker.get_recent_errors(provider=provider, limit=limit)

    if fmt == "json":
        click.echo(json_mod.dumps(errors, indent=2, default=str))
        return

    if not errors:
        click.echo("No errors recorded.")
        return

    click.echo(f"Recent Errors (showing {len(errors)} of last {limit})")
    click.echo("=" * 70)
    for i, err in enumerate(errors, 1):
        severity_colors = {
            "low": "white",
            "medium": "yellow",
            "high": "red",
            "critical": "red",
        }
        sev_color = severity_colors.get(err["severity"], "white")

        click.echo(f"\n  [{i}] {err['provider']} — {err['category']}")
        click.echo(
            f"      Severity: "
            f"{click.style(err['severity'].upper(), fg=sev_color)}"
        )
        if err["status_code"]:
            click.echo(f"      Status: {err['status_code']}")
        click.echo(f"      Retryable: {'yes' if err['retryable'] else 'no'}")
        click.echo(f"      Latency: {err['latency_ms']:.1f}ms")
        click.echo(f"      Message: {err['message'][:100]}")


@health.command("categories")
@click.option(
    "--format", "-f", "fmt",
    type=click.Choice(["text", "json"]),
    default="text",
    show_default=True,
    help="Output format.",
)
def health_categories(fmt: str):
    """Show error counts grouped by category.

    Displays how many errors occurred in each category
    (network, authentication, rate_limit, timeout, etc.).

    Example:

        claude-code-model-gateway health categories
    """
    import json as json_mod

    from src.error_handling import get_error_tracker

    tracker = get_error_tracker()
    counts = tracker.get_error_counts_by_category()

    if fmt == "json":
        click.echo(json_mod.dumps(counts, indent=2))
        return

    if not counts:
        click.echo("No errors recorded.")
        return

    click.echo("Error Counts by Category")
    click.echo("=" * 40)
    for category, count in sorted(counts.items(), key=lambda x: -x[1]):
        click.echo(f"  {category:20s} {count}")
    click.echo(f"  {'─' * 30}")
    click.echo(f"  {'TOTAL':20s} {sum(counts.values())}")


@health.command("circuit-breakers")
@click.option(
    "--format", "-f", "fmt",
    type=click.Choice(["text", "json"]),
    default="text",
    show_default=True,
    help="Output format.",
)
def health_circuit_breakers(fmt: str):
    """Show circuit breaker states.

    Lists all registered circuit breakers with their current state,
    failure counts, and configuration.

    Example:

        claude-code-model-gateway health circuit-breakers
    """
    import json as json_mod

    from src.retry import list_circuit_breakers

    breakers = list_circuit_breakers()

    if fmt == "json":
        data = {name: cb.get_stats() for name, cb in breakers.items()}
        click.echo(json_mod.dumps(data, indent=2))
        return

    if not breakers:
        click.echo("No circuit breakers registered.")
        return

    click.echo("Circuit Breakers")
    click.echo("=" * 50)
    for name, cb in breakers.items():
        stats = cb.get_stats()
        state = stats["state"]
        state_colors = {
            "closed": "green",
            "open": "red",
            "half_open": "yellow",
        }
        s_color = state_colors.get(state, "white")

        click.echo(f"\n  {name}")
        click.echo(
            f"    State: {click.style(state.upper(), fg=s_color, bold=True)}"
        )
        click.echo(
            f"    Failures: {stats['failure_count']}/{stats['failure_threshold']}"
        )
        click.echo(f"    Reset Timeout: {stats['reset_timeout']}s")


@health.command("reset")
@click.option(
    "--errors", "reset_errors",
    is_flag=True,
    help="Reset error tracking statistics.",
)
@click.option(
    "--circuit-breakers", "reset_breakers",
    is_flag=True,
    help="Reset all circuit breakers to CLOSED.",
)
@click.option(
    "--all", "reset_all",
    is_flag=True,
    help="Reset everything (errors + circuit breakers).",
)
def health_reset(reset_errors: bool, reset_breakers: bool, reset_all: bool):
    """Reset health tracking state.

    Reset error tracking statistics, circuit breaker states, or both.

    Example:

        claude-code-model-gateway health reset --all
        claude-code-model-gateway health reset --circuit-breakers
    """
    from src.error_handling import get_error_tracker
    from src.retry import reset_all_circuit_breakers

    if not any([reset_errors, reset_breakers, reset_all]):
        click.echo("Specify --errors, --circuit-breakers, or --all")
        return

    if reset_all or reset_errors:
        tracker = get_error_tracker()
        tracker.reset()
        click.echo("Error tracking statistics reset.")

    if reset_all or reset_breakers:
        reset_all_circuit_breakers()
        click.echo("All circuit breakers reset to CLOSED.")


@health.command("retry-policies")
@click.option(
    "--format", "-f", "fmt",
    type=click.Choice(["text", "json"]),
    default="text",
    show_default=True,
    help="Output format.",
)
def health_retry_policies(fmt: str):
    """Show registered retry policies.

    Lists all named retry policies with their configuration
    and adaptive state (if enabled).

    Example:

        claude-code-model-gateway health retry-policies
    """
    import json as json_mod

    from src.retry import list_retry_policies

    policies = list_retry_policies()

    if fmt == "json":
        data = {}
        for name, policy in policies.items():
            entry = {
                "name": policy.name,
                "max_attempts": policy.retry_config.max_attempts,
                "backoff_strategy": policy.retry_config.backoff_strategy.value,
                "base_delay": policy.retry_config.base_delay,
                "max_delay": policy.retry_config.max_delay,
                "jitter": policy.retry_config.jitter,
            }
            if policy.adaptive:
                entry["adaptive_state"] = policy.adaptive.get_state()
            data[name] = entry
        click.echo(json_mod.dumps(data, indent=2))
        return

    if not policies:
        click.echo("No retry policies registered.")
        return

    click.echo("Retry Policies")
    click.echo("=" * 50)
    for name, policy in policies.items():
        click.echo(f"\n  {name}")
        click.echo(f"    Max Attempts: {policy.retry_config.max_attempts}")
        click.echo(f"    Backoff: {policy.retry_config.backoff_strategy.value}")
        click.echo(f"    Base Delay: {policy.retry_config.base_delay}s")
        click.echo(f"    Max Delay: {policy.retry_config.max_delay}s")
        click.echo(f"    Jitter: {'yes' if policy.retry_config.jitter else 'no'}")
        if policy.adaptive:
            state = policy.adaptive.get_state()
            click.echo("    Adaptive: YES")
            click.echo(
                f"      Current Max Attempts: {state['current_max_attempts']}"
            )
            click.echo(
                f"      Delay Multiplier: {state['current_delay_multiplier']:.1f}x"
            )
            click.echo(f"      Strategy: {state['current_strategy']}")


# --------------------------------------------------------------------------- #
# Route management commands
# --------------------------------------------------------------------------- #


@main.group("route")
def route_group():
    """Manage API request routing rules and inspect routing behaviour.

    Configure how incoming API requests are routed to upstream providers
    based on model names, headers, or custom rules.

    Example:

        claude-code-model-gateway route list

        claude-code-model-gateway route resolve --model claude-sonnet-4-20250514

        claude-code-model-gateway route add "claude-*" --provider anthropic
    """
    pass


@route_group.command("list")
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["text", "json"]),
    default="text",
    show_default=True,
    help="Output format.",
)
@click.option(
    "--config-file",
    type=click.Path(exists=True),
    default=None,
    help="Path to the gateway configuration file.",
)
def route_list(fmt: str, config_file: Optional[str]):
    """List all configured routing rules.

    Shows the model-to-provider mapping built from the current
    configuration plus any explicit routing rules.

    Example:

        claude-code-model-gateway route list

        claude-code-model-gateway route list --format json
    """
    import json as json_mod
    from pathlib import Path as P

    from src.config import load_config
    from src.router import Router, build_model_provider_map

    config_path = P(config_file) if config_file else None
    config = load_config(path=config_path, validate=False, use_cache=False)

    router = Router.from_config(config)
    model_map = router.model_map
    rules = router.rules

    if fmt == "json":
        data = {
            "strategy": router.strategy.value,
            "fallback_provider": router.fallback_provider,
            "model_map": model_map,
            "rules": [r.to_dict() for r in rules],
        }
        click.echo(json_mod.dumps(data, indent=2))
        return

    click.echo("Routing Configuration")
    click.echo("=" * 60)
    click.echo(f"  Strategy:  {router.strategy.value}")
    click.echo(f"  Fallback:  {router.fallback_provider or '(none)'}")

    if rules:
        click.echo()
        click.echo("Explicit Rules:")
        click.echo("-" * 60)
        click.echo(f"  {'Pattern':<30} {'Provider':<15} {'Type':<15} {'Pri'}")
        click.echo(f"  {'-'*30} {'-'*15} {'-'*15} {'-'*5}")
        for rule in rules:
            click.echo(
                f"  {rule.pattern:<30} {rule.provider:<15} "
                f"{rule.rule_type.value:<15} {rule.priority}"
            )

    if model_map:
        click.echo()
        click.echo("Model -> Provider Map:")
        click.echo("-" * 60)
        for model_name in sorted(model_map.keys()):
            provider = model_map[model_name]
            click.echo(f"  {model_name:<40} -> {provider}")
    else:
        click.echo()
        click.echo("  No models configured.")


@route_group.command("resolve")
@click.option("--model", "-m", default=None, help="Model name to resolve.")
@click.option("--path", "-p", default="/v1/messages", help="Request path.")
@click.option(
    "--header",
    "-H",
    multiple=True,
    help="Request header in 'Key: Value' format.",
)
@click.option(
    "--config-file",
    type=click.Path(exists=True),
    default=None,
    help="Path to the gateway configuration file.",
)
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["text", "json"]),
    default="text",
    show_default=True,
    help="Output format.",
)
def route_resolve(
    model: Optional[str],
    path: str,
    header: tuple,
    config_file: Optional[str],
    fmt: str,
):
    """Resolve which provider would handle a request.

    Simulates routing for a given model name and/or path to show
    which provider would be selected.

    Example:

        claude-code-model-gateway route resolve --model gpt-4o

        claude-code-model-gateway route resolve --model claude-sonnet-4-20250514

        claude-code-model-gateway route resolve -H "X-Provider: anthropic"
    """
    import json as json_mod
    from pathlib import Path as P

    from src.config import load_config
    from src.router import RequestContext, Router

    config_path = P(config_file) if config_file else None
    config = load_config(path=config_path, validate=False, use_cache=False)
    router = Router.from_config(config)

    # Parse headers
    headers: dict[str, str] = {}
    for h in header:
        if ":" in h:
            key, val = h.split(":", 1)
            headers[key.strip().lower()] = val.strip()

    ctx = RequestContext(
        method="POST",
        path=path,
        model=model,
        headers=headers,
    )

    try:
        match = router.resolve(ctx)

        if fmt == "json":
            click.echo(json_mod.dumps(match.to_dict(), indent=2))
            return

        click.echo("Route Resolution")
        click.echo("=" * 50)
        click.echo(f"  Model:           {match.model or '(none)'}")
        click.echo(f"  Provider:        {match.provider_name}")
        click.echo(f"  Strategy:        {match.strategy.value}")
        click.echo(f"  Fallback used:   {match.fallback_used}")
        click.echo(f"  Resolution time: {match.resolution_time_ms:.3f}ms")
        if match.rule:
            click.echo(f"  Matched rule:    {match.rule.pattern} -> {match.rule.provider}")
        click.echo()
        click.echo("Provider Details:")
        click.echo(f"  Name:         {match.provider_config.display_name}")
        click.echo(f"  API Base:     {match.provider_config.api_base}")
        click.echo(f"  Auth Type:    {match.provider_config.auth_type.value}")
        click.echo(f"  Enabled:      {match.provider_config.enabled}")

    except Exception as exc:
        if fmt == "json":
            click.echo(
                json_mod.dumps({"error": str(exc)}, indent=2)
            )
        else:
            click.secho(f"  Error: {exc}", fg="red")
        raise SystemExit(1)


@route_group.command("add")
@click.argument("pattern")
@click.option(
    "--provider",
    required=True,
    help="Target provider name.",
)
@click.option(
    "--type",
    "rule_type",
    type=click.Choice(["model_pattern", "path_pattern", "header_match", "catch_all"]),
    default="model_pattern",
    show_default=True,
    help="Type of pattern matching.",
)
@click.option(
    "--priority",
    default=0,
    show_default=True,
    type=int,
    help="Rule priority (higher = evaluated first).",
)
@click.option(
    "--header-name",
    default="",
    help="Header name (for header_match rules).",
)
@click.option(
    "--header-value",
    default="",
    help="Expected header value (for header_match rules).",
)
@click.option(
    "--name",
    "rule_name",
    default="",
    help="Human-readable name for this rule.",
)
@click.option(
    "--config-file",
    type=click.Path(),
    default=None,
    help="Path to the gateway configuration file.",
)
def route_add(
    pattern: str,
    provider: str,
    rule_type: str,
    priority: int,
    header_name: str,
    header_value: str,
    rule_name: str,
    config_file: Optional[str],
):
    """Add a routing rule.

    PATTERN is the match pattern (glob for models, glob for paths).

    Example:

        claude-code-model-gateway route add "claude-*" --provider anthropic

        claude-code-model-gateway route add "gpt-*" --provider openai --priority 10

        claude-code-model-gateway route add "/v1/*" --provider anthropic --type path_pattern
    """
    from pathlib import Path as P

    from src.config import load_config, save_config_file
    from src.router import RouteRule, RouteRuleType

    config_path = P(config_file) if config_file else P("gateway.yaml")

    try:
        config = load_config(path=config_path if config_path.exists() else None,
                             validate=False, use_cache=False)
    except Exception:
        from src.models import GatewayConfig
        config = GatewayConfig()

    # Build the rule
    rule = RouteRule(
        pattern=pattern,
        provider=provider,
        rule_type=RouteRuleType(rule_type),
        priority=priority,
        header_name=header_name,
        header_value=header_value,
        name=rule_name,
    )

    # Add to config's extra routing_rules
    if "routing_rules" not in config.extra:
        config.extra["routing_rules"] = []
    config.extra["routing_rules"].append(rule.to_dict())

    # Save
    save_config_file(config, config_path)

    click.secho(
        f"Added route rule: {pattern} -> {provider} "
        f"(type={rule_type}, priority={priority})",
        fg="green",
    )


@route_group.command("remove")
@click.argument("pattern")
@click.option(
    "--provider",
    required=True,
    help="Target provider name to match.",
)
@click.option(
    "--config-file",
    type=click.Path(),
    default=None,
    help="Path to the gateway configuration file.",
)
def route_remove(pattern: str, provider: str, config_file: Optional[str]):
    """Remove a routing rule by pattern and provider.

    Example:

        claude-code-model-gateway route remove "claude-*" --provider anthropic
    """
    from pathlib import Path as P

    from src.config import load_config, save_config_file

    config_path = P(config_file) if config_file else P("gateway.yaml")
    config = load_config(path=config_path if config_path.exists() else None,
                         validate=False, use_cache=False)

    rules = config.extra.get("routing_rules", [])
    before = len(rules)
    rules = [
        r for r in rules
        if not (r.get("pattern") == pattern and r.get("provider") == provider)
    ]
    config.extra["routing_rules"] = rules
    removed = before - len(rules)

    save_config_file(config, config_path)

    if removed:
        click.secho(
            f"Removed {removed} rule(s): {pattern} -> {provider}",
            fg="green",
        )
    else:
        click.secho(
            f"No matching rule found: {pattern} -> {provider}",
            fg="yellow",
        )


@route_group.command("test")
@click.option("--model", "-m", multiple=True, help="Model names to test routing for.")
@click.option(
    "--config-file",
    type=click.Path(exists=True),
    default=None,
    help="Path to the gateway configuration file.",
)
def route_test(model: tuple, config_file: Optional[str]):
    """Test routing for one or more model names.

    Resolves each model and displays the routing decision in a summary
    table.

    Example:

        claude-code-model-gateway route test -m gpt-4o -m claude-sonnet-4-20250514 -m gemini-2.0-flash
    """
    from pathlib import Path as P

    from src.config import load_config
    from src.router import RequestContext, Router

    config_path = P(config_file) if config_file else None
    config = load_config(path=config_path, validate=False, use_cache=False)
    router = Router.from_config(config)

    if not model:
        # Test with all known models
        model = tuple(sorted(router.model_map.keys()))

    if not model:
        click.echo("No models to test. Configure providers with models first.")
        return

    click.echo("Route Resolution Test")
    click.echo("=" * 70)
    click.echo(f"  {'Model':<35} {'Provider':<15} {'Strategy':<15} {'Time'}")
    click.echo(f"  {'-'*35} {'-'*15} {'-'*15} {'-'*8}")

    for m in model:
        ctx = RequestContext(method="POST", path="/v1/messages", model=m)
        try:
            match = router.resolve(ctx)
            click.echo(
                f"  {m:<35} {match.provider_name:<15} "
                f"{match.strategy.value:<15} {match.resolution_time_ms:.1f}ms"
            )
        except Exception as exc:
            click.secho(f"  {m:<35} {'ERROR':<15} {str(exc)[:30]}", fg="red")

    click.echo()
    stats = router.get_stats()
    click.echo(f"Total: {stats['total_requests']} | "
               f"Rules: {stats['rule_matches']} | "
               f"Model: {stats['model_matches']} | "
               f"Strategy: {stats['strategy_matches']} | "
               f"Fallback: {stats['fallback_matches']} | "
               f"NoRoute: {stats['no_route']}")


@route_group.command("stats")
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["text", "json"]),
    default="text",
    show_default=True,
    help="Output format.",
)
@click.option(
    "--config-file",
    type=click.Path(exists=True),
    default=None,
    help="Path to the gateway configuration file.",
)
def route_stats(fmt: str, config_file: Optional[str]):
    """Show routing statistics and model mapping summary.

    Example:

        claude-code-model-gateway route stats

        claude-code-model-gateway route stats --format json
    """
    import json as json_mod
    from pathlib import Path as P

    from src.config import load_config
    from src.router import Router

    config_path = P(config_file) if config_file else None
    config = load_config(path=config_path, validate=False, use_cache=False)
    router = Router.from_config(config)

    model_map = router.model_map
    providers = config.get_enabled_providers()

    # Count models per provider
    models_per_provider: dict[str, int] = {}
    for _, prov_name in model_map.items():
        models_per_provider[prov_name] = models_per_provider.get(prov_name, 0) + 1

    data = {
        "strategy": router.strategy.value,
        "fallback_provider": router.fallback_provider,
        "total_providers": len(providers),
        "total_models": len(model_map),
        "total_rules": len(router.rules),
        "models_per_provider": models_per_provider,
    }

    if fmt == "json":
        click.echo(json_mod.dumps(data, indent=2))
        return

    click.echo("Routing Statistics")
    click.echo("=" * 50)
    click.echo(f"  Strategy:          {data['strategy']}")
    click.echo(f"  Fallback:          {data['fallback_provider'] or '(none)'}")
    click.echo(f"  Total providers:   {data['total_providers']}")
    click.echo(f"  Total models:      {data['total_models']}")
    click.echo(f"  Total rules:       {data['total_rules']}")

    if models_per_provider:
        click.echo()
        click.echo("Models per Provider:")
        click.echo("-" * 50)
        for prov_name in sorted(models_per_provider.keys()):
            count = models_per_provider[prov_name]
            click.echo(f"  {prov_name:<25} {count} model(s)")


@route_group.command("serve")
@click.option(
    "--host",
    "-H",
    default="127.0.0.1",
    show_default=True,
    help="Host address to bind to.",
)
@click.option(
    "--port",
    "-p",
    default=3001,
    show_default=True,
    type=int,
    help="Port to listen on.",
)
@click.option(
    "--timeout",
    "-t",
    default=30,
    show_default=True,
    type=int,
    help="Upstream connection timeout in seconds.",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Enable verbose (DEBUG) logging.",
)
@click.option(
    "--log-format",
    type=click.Choice(["standard", "detailed", "json", "colored", "minimal"]),
    default="standard",
    show_default=True,
    help="Log output format.",
)
@click.option(
    "--log-file",
    default=None,
    type=click.Path(dir_okay=False),
    help="Path to log file (enables file logging).",
)
@click.option(
    "--max-retries",
    default=2,
    show_default=True,
    type=int,
    help="Number of upstream retry attempts on transient errors (0 = no retries).",
)
@click.option(
    "--retry-delay",
    default=1.0,
    show_default=True,
    type=float,
    help="Base retry back-off delay in seconds (exponential backoff).",
)
@click.option(
    "--max-request-size",
    default=10485760,
    show_default=True,
    type=int,
    help="Maximum request body size in bytes.",
)
@click.option(
    "--max-rpm",
    default=0,
    show_default=True,
    type=int,
    help="Per-client rate limit in requests per minute (0 = unlimited).",
)
@click.option(
    "--no-auth",
    is_flag=True,
    default=False,
    help="Disable API key requirement (useful for development/testing).",
)
@click.option(
    "--config-file",
    "-c",
    type=click.Path(exists=True),
    default=None,
    help="Path to gateway configuration file.",
)
def route_serve(
    host: str,
    port: int,
    timeout: int,
    verbose: bool,
    log_format: str,
    log_file: Optional[str],
    max_retries: int,
    retry_delay: float,
    max_request_size: int,
    max_rpm: int,
    no_auth: bool,
    config_file: Optional[str],
):
    """Start the multi-provider routing gateway server.

    Launches a local HTTP server that routes incoming API requests to the
    appropriate upstream provider using the configured routing rules,
    model-to-provider mapping, and interceptor chain.

    The gateway accepts requests in Anthropic Messages API format, runs them
    through the interceptor chain (authentication, rate limiting, routing),
    and forwards them to the target provider.

    Features:

    \b
      - Model-based routing (route by model field in request body)
      - Header-based routing (route by X-Provider header)
      - Per-client rate limiting (--max-rpm)
      - Auth header injection from environment variables
      - Configurable retry with exponential backoff
      - Health check and status endpoints

    Internal endpoints (never forwarded upstream):

    \b
      GET  /health    - Health check
      GET  /status    - Gateway statistics and interceptor chain stats

    Example:

    \b
        claude-code-model-gateway route serve
        claude-code-model-gateway route serve --port 8080 --max-rpm 60
        claude-code-model-gateway route serve --no-auth --verbose
        claude-code-model-gateway route serve -c gateway.yaml --max-retries 3
    """
    import os
    from pathlib import Path as P

    from src.config import load_config
    from src.gateway import run_gateway
    from src.interceptor import create_default_chain

    log_level = "debug" if verbose else "info"
    output = "both" if log_file else "console"
    setup_logging(
        level=log_level,
        log_format=log_format,
        output=output,
        log_file=log_file,
    )

    logger = get_logger("cli.route.serve")

    # Load gateway configuration
    config_path = P(config_file) if config_file else None
    try:
        config = load_config(path=config_path, validate=False, use_cache=False)
    except Exception as exc:
        click.secho(f"Error loading configuration: {exc}", fg="red", err=True)
        raise SystemExit(1)

    # Summary
    enabled_providers = config.get_enabled_providers()
    click.echo(f"Starting multi-provider routing gateway on {host}:{port}")
    click.echo(f"  Providers: {len(enabled_providers)} enabled")
    if enabled_providers:
        for pname, prov in sorted(enabled_providers.items()):
            marker = " *" if pname == config.default_provider else "  "
            click.echo(f"  {marker} {prov.display_name} ({pname}) -> {prov.api_base}")
    click.echo(f"  Timeout:   {timeout}s")
    click.echo(f"  Retries:   {max_retries} (base delay: {retry_delay}s)")
    click.echo(f"  Rate limit:{f' {max_rpm} rpm/client' if max_rpm else ' disabled'}")
    click.echo(f"  Auth:      {'disabled (--no-auth)' if no_auth else 'required'}")
    click.echo()
    click.echo("Internal endpoints:")
    click.echo(f"  GET  http://{host}:{port}/health")
    click.echo(f"  GET  http://{host}:{port}/status")
    click.echo()
    logger.info(
        "Starting routing gateway on %s:%d with %d providers",
        host,
        port,
        len(enabled_providers),
    )

    run_gateway(
        config,
        host=host,
        port=port,
        upstream_timeout=timeout,
        max_request_size=max_request_size,
        max_retries=max_retries,
        retry_base_delay=retry_delay,
        max_rpm=max_rpm,
        require_auth=not no_auth,
    )


if __name__ == "__main__":
    main()
