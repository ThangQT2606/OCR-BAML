class NewsInfo {
  title string @alias("Tiêu đề bài báo") @description("Tiêu đề chính của bài báo (nếu có)")
  content string @alias("Nội dung bài báo") @description("Nội dung chi tiết của bài báo")
  language string @alias("Ngôn ngữ") @description("Ngôn ngữ của bài báo được phát hiện")
  level string @alias("Cấp độ của bài báo") @description("Trình độ của bài báo ở trong các cấp độ [N1, N2, N3, N4, N5]")
}

function ExtractNews(images: image[]) -> NewsInfo {
    client "Gemini_2_0_pro"
    prompt #"
    あなたは日本語の専門家なので、画像を解析し、記事情報を NewsInfo オブジェクトとして返します。 
    また、その記事の文法を分析し、記事のレベルを返します。
    {% for img in images %}
      image-{{loop.index}}:  
      {{ img }}
    {% endfor %}
    {{ ctx.output_format }}
    "#
}