use std::path::PathBuf;

use clap::Parser;

#[derive(Parser)]
struct Args {
    model_path: Option<PathBuf>,
    #[arg(long, short = 'v')]
    pub tokenizer_path: Option<PathBuf>,
    #[arg(long, short = 'r')]
    pub tokenizer_repository: Option<String>,
    #[arg(long)]
    pub use_gpu: Option<bool>,
}
impl Args {
    pub fn to_tokenizer_source(&self) -> llm::TokenizerSource {
        match (&self.tokenizer_path, &self.tokenizer_repository) {
            (Some(_), Some(_)) => {
                panic!("Cannot specify both --tokenizer-path and --tokenizer-repository");
            }
            (Some(path), None) => llm::TokenizerSource::HuggingFaceTokenizerFile(path.to_owned()),
            (None, Some(repo)) => llm::TokenizerSource::HuggingFaceRemote(repo.to_owned()),
            (None, None) => llm::TokenizerSource::Embedded,
        }
    }
}

#[derive(Clone)]
struct BenchResult {
    elapsed: std::time::Duration,
    query_token_count: usize,
}

impl BenchResult {
    /// number of tokens embedded per millisecond
    fn rate(&self) -> f64 {
        (self.query_token_count as f64) / (self.elapsed.as_millis() as f64)
    }
}

impl std::fmt::Display for BenchResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} tokens, {} ms",
            self.query_token_count,
            self.elapsed.as_millis(),
        )
    }
}

impl std::ops::Add<BenchResult> for BenchResult {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            elapsed: self.elapsed.add(rhs.elapsed),
            query_token_count: self.query_token_count + rhs.query_token_count,
        }
    }
}

fn main() {
    let mut args = Args::parse();
    args.tokenizer_path = Some("/Users/gabriel/work/research/rt-bench/model/tokenizer.json".into());
    args.model_path =
        Some("/Users/gabriel/work/research/rt-bench/model/ggml-model-q4_0.bin".into());

    let tokenizer_source = args.to_tokenizer_source();
    let model_architecture = llm::ModelArchitecture::Bert;
    let model_path = args.model_path.unwrap();
    let corpus = &include_str!("./vicuna-chat.rs")
        .lines()
        .map(|l| &l[..500.min(l.len())])
        .take(2)
        .collect::<Vec<_>>();

    // Load model
    let mut model_params = llm::ModelParameters::default();
    model_params.use_gpu = true;
    let model = llm::load_dynamic(
        Some(model_architecture),
        &model_path,
        tokenizer_source,
        model_params,
        llm::load_progress_callback_stdout,
    )
    .unwrap_or_else(|err| {
        panic!("Failed to load {model_architecture} model from {model_path:?}: {err}")
    });
    let inference_parameters = llm::InferenceParameters::default();

    // Generate embeddings for query and comparands
    let results = get_batch_embeddings(model.as_ref(), &inference_parameters, corpus);
}

fn get_embeddings(
    model: &dyn llm::Model,
    _inference_parameters: &llm::InferenceParameters,
    query: &str,
) -> BenchResult {
    dbg!(&query);
    let s = std::time::Instant::now();
    let session_config = llm::InferenceSessionConfig {
        ..Default::default()
    };
    let mut session = model.start_session(session_config);
    let mut output_request = llm::OutputRequest {
        all_logits: None,
        embeddings: Some(Vec::new()),
    };
    let vocab = model.tokenizer();
    let beginning_of_sentence = true;
    let query_token_ids = vocab
        .tokenize(query, beginning_of_sentence)
        .unwrap()
        .iter()
        .map(|(_, tok)| *tok)
        .collect::<Vec<_>>();
    model.evaluate(&mut session, &query_token_ids, &mut output_request);
    let _embeddings = output_request.embeddings.unwrap();
    BenchResult {
        elapsed: s.elapsed(),
        query_token_count: query_token_ids.len(),
    }
}

fn get_batch_embeddings(
    model: &dyn llm::Model,
    _inference_parameters: &llm::InferenceParameters,
    corpus: &[&str],
) -> BenchResult {
    dbg!(&corpus);
    let s = std::time::Instant::now();
    let session_config = llm::InferenceSessionConfig {
        ..Default::default()
    };
    let mut session = model.start_session(session_config);
    let mut output_request = llm::OutputRequest {
        all_logits: None,
        embeddings: Some(Vec::new()),
    };
    let vocab = model.tokenizer();
    let beginning_of_sentence = true;

    let query_token_ids = corpus
        .iter()
        .map(|q| {
            vocab
                .tokenize(q, beginning_of_sentence)
                .unwrap()
                .iter()
                .map(|(_, tok)| *tok)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let query_token_ids: Vec<_> = query_token_ids.iter().map(AsRef::as_ref).collect();

    model.batch_evaluate(&mut session, &query_token_ids, &mut output_request);
    let _embeddings = output_request.embeddings.unwrap();
    dbg!(&_embeddings);

    BenchResult {
        elapsed: s.elapsed(),
        query_token_count: query_token_ids.len(),
    }
}
