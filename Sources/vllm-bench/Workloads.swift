import Foundation

// Prefill-heavy "blog research" workload modeled on wsearch agentic calls:
// a research-agent system prompt + several long travel-blog excerpts + an
// extraction question answered as compact JSON. Prompt size is controlled by
// how many excerpts are stacked; requests rotate excerpts/questions so
// sequences don't share a KV prefix beyond the system prompt.

public enum PromptPreset: String, CaseIterable, Identifiable, Codable {
    case short = "Short (~600 tok)"
    case medium = "Medium (~1.6K tok)"
    case long = "Long (~3K tok)"

    public var id: String { rawValue }

    public var excerptCount: Int {
        switch self {
        case .short: return 1
        case .medium: return 4
        case .long: return 8
        }
    }
}

public enum Workloads {

    public static let systemPrompt = """
    You are a travel research agent. You are given excerpts from travel blogs that were \
    retrieved for a research question. Read every excerpt carefully and extract only facts \
    that are stated in the text. Respond with a single JSON object with these keys: \
    "answer" (a two-sentence synthesis grounded in the excerpts), "places" (an array of \
    objects with "name", "city", and "why" fields for every specific place the excerpts \
    recommend that is relevant to the question), "tips" (an array of short practical tips \
    stated in the excerpts, such as timing, tickets, or transport), and "confidence" \
    (one of "high", "medium", "low" based on how directly the excerpts address the \
    question). Do not invent places that are not mentioned. Do not add commentary outside \
    the JSON object.
    """

    public static let questions = [
        "Which neighborhoods do these bloggers recommend for a first visit, and why?",
        "What food experiences do the excerpts recommend, and where exactly are they?",
        "What practical timing and ticket advice do the excerpts give for the main sights?",
        "Which day trips or side excursions do the bloggers describe as worth the effort?",
        "What do the excerpts say about getting around: transit passes, walking, cycling?",
        "Which places do the bloggers describe as overrated or skippable, and what do they suggest instead?",
    ]

    public static let excerpts: [String] = [
        """
        [Blog: Slow Mornings in Kyoto — day 3] We left the machiya before sunrise and walked \
        the empty stretch of Ninenzaka while the lanterns were still lit, and it was the only \
        time all week the lane felt like the photographs. By eight the tour groups had filled \
        the slope up to Kiyomizu-dera, so my honest advice is to be at the temple gate at six \
        when it opens, do the veranda first, and take the Otani cemetery path down instead of \
        retracing the shopping streets. We paid four hundred yen each and spent almost two \
        hours there. Afterwards we had the tamago sando and drip coffee at a kissaten near \
        Rokuhara that opens at seven, which saved us from the cafe queues that form later on \
        Matsubara-dori. If you only take one thing from this post: the famous spots are \
        genuinely wonderful, but they are wonderful at opening time and something else \
        entirely at eleven. Book the earliest slot for everything, then nap.
        """,
        """
        [Blog: Rails and Ramen — Tokyo on a transit pass] Everyone asks whether the tourist \
        rail passes pay off inside Tokyo, and for us the answer was no: we loaded a plain IC \
        card, kept our rides under a thousand yen most days, and never once queued at a fare \
        machine. The pattern that worked was picking one hub per day — Ueno one day, Shibuya \
        the next — and doing everything within a thirty-minute walking radius before touching \
        a train again. The Yamanote line is the spine, but the quiet wins were the small \
        private lines: the Setagaya tramway took us past plant nurseries and shrine gardens \
        no guidebook mentioned, and the Toden Arakawa line does the same for the north side. \
        Late evening is the best riding time; after nine the platforms empty out and you can \
        actually see the city from the elevated sections. Skip the taxi apps unless it is \
        raining; the surge pricing after midnight is real and the trains restart at five.
        """,
        """
        [Blog: A Week of Small Plates — San Sebastián] The pintxos crawl advice you read \
        everywhere is directionally right and specifically wrong. Yes, do one plate and one \
        drink per bar and keep moving; no, do not start on the main drag of the Parte Vieja \
        at eight, because that is when the cruise crowds arrive. We started at six-thirty in \
        Gros, across the river, where the counters at the neighborhood bars were fully loaded \
        and the bartenders had time to tell us what was just out of the kitchen. The grilled \
        foie at a corner bar on Calle Zabaleta and the beef cheek at the place by the surf \
        beach were the two best bites of the entire trip, and neither appears on any list I \
        have seen. Cross into the old town at nine when the locals do. Order the txakoli and \
        let them pour it; asking for a bottle marks you instantly. Budget thirty euros a head \
        and you will eat like royalty.
        """,
        """
        [Blog: Fjord Roads — driving the Norwegian west coast] The single most useful thing \
        we learned: the ferries are the schedule, not the roads. Distances that look like two \
        hours on the map become four when you miss a crossing by five minutes, so we started \
        planning our days around the ferry timetable app and everything relaxed. The \
        Trollstigen road did not open until late May the year we went; check the seasonal \
        closures before promising anyone hairpin photos. Geiranger at midday in July is a \
        cruise port, full stop — we slept in Valldal, drove in at seven, had the viewpoints \
        to ourselves, and were leaving as the buses arrived. Fuel is expensive but stations \
        are frequent until you leave the E39; north of Ålesund fill up when you can. The \
        supermarket chain bakeries are the road-trip hack nobody mentions: fresh skillingsboller \
        and drinkable coffee for a tenth of the cafe price, in every town with a roundabout.
        """,
        """
        [Blog: Two Wheels, One Island — cycling Taiwan's east coast] Route 11 between Hualien \
        and Taitung is the ride I now measure every other ride against: the Pacific on your \
        left shoulder for a hundred and seventy kilometers, a tailwind most mornings if you \
        ride north to south, and a 7-Eleven with cold oolong every twenty minutes. The climb \
        out of the rift valley to the Tropic of Cancer marker is the only sustained effort, \
        and the tunnel section south of Baqi has a marked cycle lane but bring a rear light \
        anyway. We shipped our bags ahead with the hotel-to-hotel luggage service and rode \
        with day packs, which turned a hard tour into a holiday. In the small towns the \
        breakfast shops close by ten; eat early or you are on convenience store onigiri until \
        lunch. The train stations along the line take bikes in a bag, so if a typhoon window \
        closes in you can bail out from almost anywhere — we did exactly that on the last day \
        and regretted nothing.
        """,
        """
        [Blog: Museum Legs — an art week in Paris that skipped the queues] The Louvre \
        strategy that finally worked for us, on the third attempt in ten years: Wednesday \
        night opening, entry booked for six pm, straight to the Richelieu wing while the \
        entire planet stands in front of one painting in Denon. We had the Cour Marly \
        sculptures essentially alone. The Orangerie at opening on a weekday needs no tricks; \
        the Monet rooms are quiet for the first forty minutes. The Musée Jacquemart-André \
        was the sleeper hit of the week, a mansion collection you can see fully in ninety \
        minutes with a spectacular tearoom for lunch. Skip the top of the Eiffel Tower if \
        the forecast is hazy; the second level is better for photographs anyway, and the \
        view from the Printemps rooftop terrace is free, uncrowded, and includes the tower \
        itself, which the tower view famously does not. Buy every ticket online, always, \
        even same-day from the sidewalk outside.
        """,
        """
        [Blog: Highlands Without a Car — Scotland by bus and boot] We did nine days from \
        Inverness with nothing but a Citylink pass, local buses, and boots. The spine route \
        down the Great Glen runs often enough that missing a bus costs an hour, not a day. \
        Fort William is the logical base for Glen Nevis and the Nevis Range gondola, but we \
        preferred three nights in Kinlochleven, a village the West Highland Way walks \
        straight through: the two pubs there feed hikers properly and the Grey Mare's Tail \
        waterfall is twenty minutes from the door. The Glencoe valley buses will drop you at \
        the trailheads if you ask the driver; the Lost Valley scramble took us three hours \
        return in good weather and was the best half day of the trip. Book Skye accommodation \
        further ahead than you think reasonable — we settled for Broadford because Portree \
        was gone four months out, and honestly the quieter base suited the buses better.
        """,
        """
        [Blog: Heat, Salt, Stone — a slow week in Sicily's southeast] Ortigia is the base; \
        everything else is a day trip. The morning market on Via de Benedictis is the best \
        breakfast in Italy: pane cunzato assembled in front of you, blood orange juice, and \
        a wedge of pistachio cake for the walk back along the sea wall. Noto is forty \
        minutes by bus and needs three hours, ideally ending with granita at the cafe every \
        guide names — it deserves the fame, order the almond with a brioche and nobody will \
        judge you. Modica and Ragusa Ibla want a full day together; the bus connection \
        through the gorge is spectacular and the chocolate tastings in Modica are genuinely \
        free. The beaches south of Noto, Calamosche especially, involve a hot twenty-minute \
        walk from the parking area and are worth every step in June, but by August the \
        seagrass and the crowds arrive together. Swim early, museum at noon, granita always.
        """,
    ]

    /// Builds `count` distinct user prompts for a preset, rotating excerpts and
    /// questions so requests differ beyond the shared system prompt.
    public static func buildUserPrompts(preset: PromptPreset, count: Int) -> [String] {
        (0..<count).map { i in
            let n = preset.excerptCount
            let picked = (0..<n).map { k in excerpts[(i + k) % excerpts.count] }
            let question = questions[i % questions.count]
            var body = "Research request #\(i + 1)\n\nRetrieved blog excerpts:\n\n"
            for (j, ex) in picked.enumerated() {
                body += "--- Excerpt \(j + 1) ---\n\(ex)\n\n"
            }
            body += "Question: \(question)\n\nRespond with the JSON object only."
            return body
        }
    }
}
