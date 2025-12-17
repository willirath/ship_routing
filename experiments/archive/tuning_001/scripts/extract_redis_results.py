#!/usr/bin/env python
"""Extract all results from Redis and save to msgpack or convert to other formats."""

import click
import redis
import msgpack
from pathlib import Path


@click.command()
@click.option("--redis-host", required=True, help="Redis host address.")
@click.option("--redis-port", type=int, default=6379, help="Redis port.")
@click.option(
    "--redis-password", type=str, default=None, help="Redis password (optional)."
)
@click.option(
    "--output", type=click.Path(), default="results.msgpack", help="Output file path."
)
@click.option(
    "--output-format",
    type=click.Choice(["msgpack", "json"]),
    default="msgpack",
    help="Output format.",
)
def extract_results(redis_host, redis_port, redis_password, output, output_format):
    """Extract all results from Redis and save to file."""
    click.echo(f"Connecting to Redis at {redis_host}:{redis_port}...")
    r = redis.Redis(
        host=redis_host,
        port=redis_port,
        password=redis_password,
        decode_responses=False,
    )

    # Test connection
    try:
        r.ping()
    except Exception as e:
        click.echo(f"Error: Could not connect to Redis: {e}", err=True)
        raise

    # Get all result keys
    keys = r.keys("result:*")
    click.echo(f"Found {len(keys)} results in Redis")

    if len(keys) == 0:
        click.echo("No results found in Redis.", err=True)
        return

    # Extract all results
    results = {}
    for i, key in enumerate(keys):
        if (i + 1) % max(1, len(keys) // 10) == 0:
            click.echo(f"  Extracted {i + 1}/{len(keys)} results...", err=True)
        results[key.decode()] = r.get(key)

    click.echo(f"  Extracted all {len(keys)} results.")

    # Save based on format
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_format == "msgpack":
        with open(output_path, "wb") as f:
            msgpack.pack(results, f)
        click.echo(
            f"Results saved to {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)"
        )
    elif output_format == "json":
        import json

        json_results = {k: msgpack.unpackb(v, raw=False) for k, v in results.items()}
        with open(output_path, "w") as f:
            json.dump(json_results, f, indent=2)
        click.echo(
            f"Results saved to {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)"
        )


if __name__ == "__main__":
    extract_results()
