package ma.emsi.Chidoub.test4;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentParser;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.segment.TextSegment;

import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiEmbeddingModel;

import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;

import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;

import dev.langchain4j.rag.query.Query;
import dev.langchain4j.rag.query.router.QueryRouter;

import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;

import dev.langchain4j.model.input.Prompt;
import dev.langchain4j.model.input.PromptTemplate;


import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.nio.file.Paths;
import java.time.Duration;
import java.util.*;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

public class Test4 {

    public static void main(String[] args) {

        System.out.println("\n--- TEST 4 : Routage \"RAG ou pas\" ---\n");

        configureLogger();

        // 1) Clé API
        String apiKey = System.getenv("GEMINI_API_KEY");
        if (apiKey == null) {
            throw new IllegalStateException("⚠️ Variable d'environnement GEMINI_API_KEY manquante");
        }

        // 2) Chargement du SEUL PDF (support de cours RAG)
        DocumentParser parser = new ApacheTikaDocumentParser();
        // adapte le chemin si besoin (par ex. "src/main/resources/rag.pdf")
        Document document = FileSystemDocumentLoader.loadDocument(
                Paths.get("src/main/resources/rag.pdf"), parser);

        // 3) Découpage en segments
        List<TextSegment> segments = DocumentSplitters
                .recursive(200, 40)
                .split(document);

        // 4) Embeddings (text-embedding-004)
        EmbeddingModel embeddingModel = GoogleAiEmbeddingModel.builder()
                .apiKey(apiKey)
                .modelName("text-embedding-004")
                .build();

        List<Embedding> vectors = embeddingModel.embedAll(segments).content();

        EmbeddingStore<TextSegment> store = new InMemoryEmbeddingStore<>();
        store.addAll(vectors, segments);

        // 5) ContentRetriever pour CE PDF
        ContentRetriever retriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(store)
                .embeddingModel(embeddingModel)
                .maxResults(5)
                .minScore(0.2)
                .build();

        // 6) Modèle Gemini (servira à la fois pour les réponses ET pour décider RAG ou pas)
        ChatModel llm = GoogleAiGeminiChatModel.builder()
                .apiKey(apiKey)
                .modelName("gemini-2.5-flash")
                .temperature(0.2)
                .timeout(Duration.ofSeconds(60))
                .logRequestsAndResponses(true)
                .build();

        // 7) QueryRouter personnalisé : décider s'il faut utiliser le RAG ou pas
        //    Classe interne à main, comme demandé dans l’énoncé.
        // Classe interne à main – version compatible avec ta version de LangChain4j
        class IaQueryRouter implements QueryRouter {

            private final ChatModel routerModel;
            private final ContentRetriever ragRetriever;
            private final PromptTemplate template;

            IaQueryRouter(ChatModel routerModel, ContentRetriever ragRetriever) {
                this.routerModel = routerModel;
                this.ragRetriever = ragRetriever;
                this.template = PromptTemplate.from(
                        "Est-ce que la requête suivante porte sur l'intelligence artificielle " +
                                "ou sur le RAG ? Réponds seulement par 'oui', 'non' ou 'peut-être'.\n\n" +
                                "Requête : {{question}}"
                );
            }

            @Override
            public Collection<ContentRetriever> route(Query query) {

                // Construire le prompt depuis le template
                Map<String, Object> vars = Map.of("question", query.text());
                Prompt prompt = template.apply(vars);

                // ⚠️ Dans ta version : chat(String), pas generate()
                String answer = routerModel.chat(prompt.text());
                String normalized = answer.toLowerCase(Locale.ROOT).trim();

                System.out.println("[Router] Réponse du modèle pour la détection IA : " + normalized);

                // Si ce n'est PAS une question IA → pas de RAG
                if (normalized.startsWith("non")) {
                    return Collections.emptyList();
                }

                // Sinon → activer le RAG
                return List.of(ragRetriever);
            }
        }


        QueryRouter router = new IaQueryRouter(llm, retriever);

        // 8) RetrievalAugmentor qui utilise ce QueryRouter
        RetrievalAugmentor augmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(router)
                .build();

        // 9) Assistant RAG, comme au test précédent, mais avec ce nouvel augmentor
        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(llm)
                .retrievalAugmentor(augmentor)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .build();

        // 10) Boucle interactive de test
        try (Scanner sc = new Scanner(System.in)) {
            System.out.println("D'abord, tape 'Bonjour', puis une question sur le RAG.");
            System.out.println("Tape 'exit' pour quitter.\n");

            while (true) {
                System.out.print("> ");
                String q = sc.nextLine().trim();
                if (q.equalsIgnoreCase("exit")) break;
                if (q.isEmpty()) continue;

                String answer = assistant.answer(q);
                System.out.println("\n🤖 " + answer + "\n");
            }
        }
    }

    private static void configureLogger() {
        Logger log = Logger.getLogger("dev.langchain4j");
        log.setLevel(Level.FINE);

        ConsoleHandler handler = new ConsoleHandler();
        handler.setLevel(Level.FINE);

        if (log.getHandlers().length == 0) {
            log.addHandler(handler);
        }
    }
}
