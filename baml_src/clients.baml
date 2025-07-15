// Learn more about clients at https://docs.boundaryml.com/docs/snippets/clients/overview

client<llm> Gemini_2_0_pro {
  provider google-ai
  options {
    model "gemini-2.0-pro"
    api_key env.GOOGLE_API_KEY
    generationConfig {
      temperature 0.5
    }
  }
}

client<llm> Gemini_2_0_flash {
  provider google-ai
  options {
    model "gemini-2.0-flash"
    api_key env.GOOGLE_API_KEY
    generationConfig {
      temperature 0.5
    }
  }
}

client<llm> Gemini_2_5_pro {
  provider google-ai
  options {
    model "gemini-2.5-pro"
    api_key env.GOOGLE_API_KEY
    generationConfig {
      temperature 0.5
    }
  }
}

client<llm> Vertex {
  provider vertex-ai
  options {
    model gemini-1.5-pro
    location us-central1
    api_key env.GOOGLE_API_KEY
    generationConfig {
      temperature 0.5
    }
  }
}

client<llm> CustomGPT41 {
  provider openai
  options {
    model "gpt-4.1"
    api_key env.OPENAI_API_KEY
    temperature 0.2
  }
}

client<llm> CustomOllama {
  provider "openai-generic"
  options {
    base_url "http://Ollama_IP:Ollama_PORT/v1"
    model "llava:7b"
    temperature 0.2
  }
}

client<llm> CustomSonnet {
  provider anthropic
  options {
    model "claude-3-5-sonnet-20241022"
    api_key env.ANTHROPIC_API_KEY
    temperature 0.2
  }
}

