"""CLI commands for claude-code-model-gateway."""

import logging
from pathlib import Path
from typing import Optional

import click

from src import __version__
from src.logging_config import (
    LogFormat,
    LoggingConfig,
    get_logger,
    get_logging_status,
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
):
    """Start the Anthropic API pass-through gateway.

    Launches a local HTTP server that acts as a transparent reverse proxy
    to the Anthropic Messages API. All requests are forwarded directly to
    api.anthropic.com with proper authentication headers.

    Supported endpoints:

    \b
      POST /v1/messages           - Create a message
      POST /v1/messages/count_tokens - Count tokens
      GET  /v1/models             - List available models

    The gateway reads the API key from --api-key, the client's x-api-key
    header, or the ANTHROPIC_API_KEY environment variable (in that order).

    Both streaming (SSE) and non-streaming responses are fully supported.
    Non-streaming requests are automatically retried on transient failures
    (429, 5xx) with exponential backoff.  Streaming requests are NOT
    retried because partial SSE data may have already been sent.

    Example:

    \b
        claude-code-model-gateway gateway
        claude-code-model-gateway gateway --port 9090
        claude-code-model-gateway gateway --api-key sk-ant-...
        claude-code-model-gateway gateway -v --timeout 600
        claude-code-model-gateway gateway --max-retries 5 --retry-delay 2.0
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

    key_source = "provided" if api_key else "not set (will check per-request)"
    click.echo(f"Starting Anthropic pass-through gateway on {host}:{port}")
    click.echo(f"  Upstream:  https://api.anthropic.com/v1")
    click.echo(f"  API key:   {key_source}")
    click.echo(f"  Version:   {anthropic_version}")
    click.echo(f"  Timeout:   {timeout}s")
    click.echo(f"  Retries:   {max_retries} (base delay: {retry_delay}s)")
    click.echo()
    click.echo("Endpoints:")
    click.echo(f"  POST http://{host}:{port}/v1/messages")
    click.echo(f"  POST http://{host}:{port}/v1/messages/count_tokens")
    click.echo(f"  GET  http://{host}:{port}/v1/models")
    click.echo()

    run_passthrough(
        host=host,
        port=port,
        timeout=timeout,
        api_key=api_key,
        anthropic_version=anthropic_version,
        max_retries=max_retries,
        retry_base_delay=retry_delay,
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
    type=click.Choice(["openai", "anthropic", "azure", "google", "bedrock"]),
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


if __name__ == "__main__":
    main()
