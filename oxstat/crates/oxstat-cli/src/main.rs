use anyhow::Result;
use clap::Parser;

/// OxStat: Open-source statistical analysis suite.
#[derive(Parser, Debug)]
#[command(name = "oxstat", version, about = "Open-source SPSS alternative")]
struct Cli {
    /// SPSS syntax file to execute.
    #[arg(short, long)]
    syntax: Option<String>,

    /// Output format: text, html, json.
    #[arg(short, long, default_value = "text")]
    format: String,

    /// Input data file (.sav, .csv, .xlsx, .parquet).
    #[arg(short, long)]
    data: Option<String>,

    /// Start interactive REPL.
    #[arg(long)]
    repl: bool,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let cli = Cli::parse();

    if cli.repl {
        println!("OxStat REPL — type SPSS syntax, 'QUIT.' to exit.");
        // TODO: REPL loop
        return Ok(());
    }

    if let Some(syntax_file) = &cli.syntax {
        println!("Executing syntax file: {syntax_file}");
        // TODO: Parse and execute syntax file
        return Ok(());
    }

    // No arguments — show help
    println!("Use --help for usage. Try --repl for interactive mode.");
    Ok(())
}
