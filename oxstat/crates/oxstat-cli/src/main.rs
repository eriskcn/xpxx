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

    if let Some(data_file) = &cli.data {
        let dataset = if data_file.ends_with(".xlsx") || data_file.ends_with(".xls") {
            oxstat_io::read_excel(data_file, None)
                .map_err(|e| anyhow::anyhow!("Failed to read Excel: {e}"))?
        } else if data_file.ends_with(".parquet") {
            oxstat_io::read_parquet(data_file)
                .map_err(|e| anyhow::anyhow!("Failed to read Parquet: {e}"))?
        } else {
            oxstat_io::read_csv(data_file)
                .map_err(|e| anyhow::anyhow!("Failed to read CSV: {e}"))?
        };
        let options = oxstat_stats::descriptives::DescriptivesOptions::default();
        let output = oxstat_stats::descriptives::run(&dataset, &options);
        let rendered = oxstat_output::render_text(&output);
        println!("{rendered}");
        return Ok(());
    }

    // No arguments — show help
    println!("Use --help for usage. Try --repl for interactive mode.");
    Ok(())
}
