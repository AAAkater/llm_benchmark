text = """
    Thinking Process:\n\n1.  **Analyze the Request:**\n    *   Input text: A short paragraph about the iPhone\'s launch in 2007, its initial reception, its global impact 8 years later, and a prediction about its future role as a "true personal computer."\n    *   Task: Generate a short summary (摘要).\n    *   Language: Chinese.\n\n2.  **Analyze the Input Text:**\n    *   Sentence 1: 2007 年乔布斯向人们展示 iPhone 并宣称“它将会改变世界”，还有人认为他在夸大其词 (In 2007, Jobs showed iPhone and claimed it would change the world, some thought he was exaggerating).\n    *   Sentence 2: 然而在 8 年后，以 iPhone 为代表的触屏智能手机已经席卷全球各个角落 (However, 8 years later, touch-screen smartphones represented by iPhone have swept every corner of the globe).\n    *   Sentence 3: 未来，智能手机将会成为“真正的个人电脑”，为人类发展做出更大的贡献 (In the future, smartphones will become "true personal computers" and contribute more to human development).\n\n3.  **Identify Key Information:**\n    *   Event: iPhone launch (2007).\n    *   Initial reaction: Skepticism vs. Jobs\' claim.\n    *   Outcome (8 years later): Global success/dominance.\n    *   Future prediction: Smartphone as true PC, greater contribution.\n\n4.  **Drafting the Summary (Iterative Process):**\n    *   *Draft 1 (Too detailed):* 2007 年乔布斯发布 iPhone 说会改变世界，有人不信。但 8 年后智能手机席卷全球。未来智能手机会成为真正的个人电脑，贡献更大。\n    *   *Draft 2 (More concise):* 乔布斯 2007 年发布 iPhone 虽遭质疑，但 8 年后智能手机已风靡全球。未来它将演变为真正的个人电脑，推动人类发展。\n    *   *Draft 3 (Focusing on core message):* 2007 年 iPhone 发布虽曾被质疑，但 8 年后已席卷全球。未来智能手机将成为真正的个人电脑，助力人类发展。\n\n5.  **Refining for Brevity and Flow:**\n    *   The core narrative is: Launch -> Skepticism -> Success -> Future Potential.\n    *   Let\'s combine for maximum impact.\n    *   "2007 年 iPhone 发布虽遭质疑，但 8 年后已席卷全球，未来有望成为真正的个人电脑并推动人类发展。" (2007 iPhone launch was questioned, but 8 years later swept global, future expected to become true PC and push human development.)\n\n6.  **Final Polish (Checking against "Short Summary" requirement):**\n    *   Needs to be very concise.\n    *   "2007 年 iPhone 发布虽曾被质疑，但 8 年后已风靡全球。未来智能手机将成为真正的个人电脑，为人类发展做出更大贡献。"\n\n    *   Even shorter:\n    *   "iPhone 发布 8 年后已席卷全球，未来智能手机将成为真正的个人电脑，助力人类发展。" (Omitting the skepticism part might lose some context, but keeps it very short. However, the contrast is important.)\n\n    *   Let\'s keep the contrast briefly.\n    *   "乔布斯 2007 年发布 iPhone 虽遭质疑，但 8 年后已风靡全球。未来智能手机将演变为真正的个人电脑，推动人类发展。"\n\n7.  **Final Selection:**\n    2007 年 iPhone 发布虽曾被质疑，但 8 年后已席卷全球。未来智能手机将成为真正的个人电脑，为人类发展做出更大贡献。\n\n    Or even more compact:\n    iPhone 发布 8 年后已席卷全球，未来将成为真正的个人电脑，推动人类发展。\n\n    Let\'s go with a balanced version that captures the timeline and prediction.\n\n    *Selected:* 2007 年 iPhone 发布虽遭质疑，但 8 年后已席卷全球。未来智能手机将成为真正的个人电脑，助力人类发展。\n\n8.  **Final Check:** Does it cover the main points? Yes (Launch/Claim, Success, Future). Is it short? Yes.\n\n    *Self-Correction:* The prompt asks for a "short summary". I should ensure it flows well.\n    "2007 年乔布斯发布 iPhone 虽曾被质疑，但 8 年后智能手机已席卷全球。未来，它将成为真正的
"""


def postprocess(text: str) -> str:
    """Postprocess LCSTS model output.

    Cleans up common artifacts in generated summaries:
    - Extracts content from thinking tags
    - Takes only the last line (actual content after thinking)
    - Removes common hallucination tokens (Russian word "своей")
    - Strips list prefixes and punctuation
    """
    # First extract content from thinking tags
    text = text.strip().replace("своей", "").strip()
    # Take the last line (actual content after thinking process)
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    text = lines[-1] if lines else ""
    text = text.replace("1. ", "") if text.startswith("1. ") else text
    text = text.replace("- ", "") if text.startswith("- ") else text
    text = text.strip('"，。！')
    return text


aaa = postprocess(text)
print(aaa)
