from gradio_app import GRADIO

def main():
    # OpenAI API 키 설정
    api_key = "EMPTY"
    
    # Gradio 앱 초기화 및 실행
    app = GRADIO(api_key=api_key)
    
    try:
        # 디버그 모드로 실행 (자동 리로드 활성화)
        app.demo.launch(
            debug=True,
            share=False,  # Colab 등에서 실행할 때는 True로 변경
            server_name="0.0.0.0",
            server_port=7860,
            show_api=False,
            favicon_path=None
        )
    except KeyboardInterrupt:
        print("\n서버를 종료합니다...")
    except Exception as e:
        print(f"에러 발생: {str(e)}")
    finally:
        print("프로그램을 종료합니다.")

if __name__ == "__main__":
    main()