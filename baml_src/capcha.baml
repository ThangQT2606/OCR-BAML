class CapchaInfor {
  content string @alias("Nội dung của capcha") @description("Nội dung của capcha")
}

// function ExtractCapcha(images: image[]) -> CapchaInfor {
//     client "Gemini_2_0_pro"
//     prompt #"
//     You are a captcha expert. Please extract the captcha information from the image. Pay attention to the overlapping letters.
//     {% for img in images %}
//       image-{{loop.index}}:  
//       {{ img }}
//     {% endfor %}
//     {{ ctx.output_format }}
//     "#
// }

function ExtractCapcha(images: image[]) -> CapchaInfor {
  client "Gemini_2_0_flash"
  prompt #"
    You are a CAPTCHA recognition expert.
    Your task is to extract **exactly 5 characters** from each image of a CAPTCHA. The characters may include uppercase letters (A-Z), lower (a-z) and digits (0-9).  
    The images may contain:
    - Overlapping or touching characters.
    - Deformed or curved characters.
    - Noise lines, background grids, or occlusion.
    - Variations in thickness or partial visibility.
    ### Important Instructions:
    1. The captcha code is only 5 characters, so you are only allowed to return the 5 characters with the highest recognition accuracy.
    2. If part of a character is hard to see, use visual context clues (such as the shape, spacing, or surrounding letters) to infer it.
    3. Characters may vary in style but will remain in the allowed set: `A-Z`, `a - z` and `0-9`. No symbols.
    4. Do not explain your answer — just return the extracted code.
    5. Characters are often confused with each other, so be careful to deduce the final character.
    {% for img in images %}
    Image {{loop.index}}:
    {{ img }}
    {% endfor %}
    Return the result using the following structure:
    {{ ctx.output_format }}
  "#
}

test TestCapcha{
  functions [ExtractCapcha]
  args {
    images [
      { url "https://stc.hnammobilecare.com/hcare/uploads/images/ma-captcha-la-gi-vi-sao-nen-thiet-lap-ma-captcha-cho-website-21588517023.png" }
    ]
  }
}
